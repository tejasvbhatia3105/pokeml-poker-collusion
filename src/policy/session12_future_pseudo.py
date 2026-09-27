\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
os.environ.setdefault('POKER_PARTITION_ROOT','artifacts/compact')
from pathlib import Path
import sys
sys.path.insert(0,'src')
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from features import build as basic
from detail_features import build as detail
from sequence_features import augment
import build_outcome_roles as BOR,build_relationship_evidence as BRE
from session8_data import hand_data,targets
from session6_priority_model import load_models as cats
from session7_model import load_models as events
ROOT=Path('artifacts/evidence_session12/future_pseudo');C=pl.col
CONFIG={'positive_agreement':.9,'negative_agreement':.02,'pseudo_weight_mass_fraction':.25,'soft_targets':'arithmetic average of archived Cat and HGB event heads, both excluding outer fold','rows':'later 2,000 hands of outer-training publicly positive relationships only; no future labels','student':'same Cat seed/400 iterations/depth5/lr.035/l2=8; soft targets duplicated with fractional class weights','selection':'thresholds fixed before evaluation; no validation checkpoint choice'}
SCRATCH=Path('cache/hand_rows2')
def future():
 path=ROOT/'future_features.parquet'
 if path.exists():return pl.read_parquet(path)
 cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];labs=pl.read_csv('data/development_labels.csv').filter(C('label')==1).select('pair_id','player_1','player_2','behavior_family');players=labs.drop('behavior_family');folds=json.load(open('artifacts/policy/table_folds.json'));parts=[];start=time.time()
 for i,path0 in enumerate(sorted(Path('artifacts/detail_features').glob('T*.parquet'))):
  existing=pl.read_parquet(path0,columns=['pair_id']).join(labs.select('pair_id'),on='pair_id')
  if not len(existing):continue
  raw=basic(path0.stem,players.head(0),players)
  if raw is None or not len(raw):continue
  d=detail(path0.stem,base=raw,pairs=players).join(labs.select('pair_id','behavior_family'),on='pair_id').with_columns(((C('time')-.6)/.4).alias('relative_time'));h=pl.read_parquet(SCRATCH/path0.name).select(list(pl.read_parquet_schema(Path('artifacts/policy/hand_features')/path0.name))).filter(C('phase')=='evaluation').drop('pair_id').join(players,on=['player_1','player_2']).sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')];h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')];h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']];n=len(d);d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id'],validate='1:1');assert len(d)==n;d,_=augment(d);query=d.select('pair_id','hand_id').join(players,on='pair_id');d=d.join(BOR.build(path0.stem,query),on=['pair_id','hand_id']).join(BRE.build(path0.stem,query),on=['pair_id','hand_id']);parts.append(d.select('pair_id','hand_id','behavior_family','phase',*[C(c).cast(pl.Float32) for c in cols]).with_columns(pl.lit(folds[path0.stem]).alias('fold')))
  if i%40==0:print('future features',i,round(time.time()-start,1),flush=True)
 out=pl.concat(parts);missing=sorted(set(labs['pair_id'])-set(out['pair_id']));(ROOT/'future_coverage.json').write_text(json.dumps({'covered_pairs':out['pair_id'].n_unique(),'missing_pairs':missing,'note':'Only pairs with observed shared future hands can enter pseudo training'},indent=2));assert set(out['pair_id'])<=set(labs['pair_id']);assert out['phase'].unique().to_list()==['evaluation'];assert np.isfinite(out.select(cols).to_numpy()).all();out.write_parquet(path,compression='zstd');return out

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));futuredata=future();d=hand_data();cm,cols=cats('priority_ordered');hm,_=events('hist_eventblend');X=d.select(cols).to_numpy();FX=futuredata.select(cols).to_numpy();nf=futuredata['fold'].to_numpy();ff=futuredata['behavior_family'].to_numpy();pred=np.zeros((len(d),2));audit=[];start=time.time()
 with threadpool_limits(limits=3):
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    a,b,ea,eb,va=targets(d,f,fam);ni=np.flatnonzero((nf!=f)&(ff==fam));assert not np.any(nf[ni]==f)
    for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
     cp=cm[fam][f][k-1].predict_proba(FX[ni],thread_count=3)[:,1];hp=hm[fam][f][1][k-1].predict_proba(FX[ni])[:,1];use=((cp>.9)&(hp>.9))|((cp<.02)&(hp<.02));px=FX[ni[use]];pr=.5*(cp[use]+hp[use]);ix=np.flatnonzero(e);assert not np.any(e&va);total=.25*len(ix);weights=total/max(1,len(px));trainX=np.concatenate([X[ix],px,px]);trainY=np.r_[y[ix].astype(int),np.zeros(len(px),int),np.ones(len(px),int)];w=np.r_[np.ones(len(ix)),weights*(1-pr),weights*pr];m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=3,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);path=ROOT/f'event{k}_{fam}_fold{f}.cbm'
     if path.exists():m.load_model(str(path))
     else:m.fit(trainX,trainY,sample_weight=w);m.save_model(str(path))
     pred[va,k-1]=m.predict_proba(X[va],thread_count=3)[:,1];audit.append({'fold':f,'family':fam,'event':k,'original_eligible_hands':len(ix),'original_positive_events':int(y[ix].sum()),'future_hands_available':len(ni),'future_hands_selected':int(use.sum()),'future_positive_agreement':int(((cp>.9)&(hp>.9)).sum()),'future_negative_agreement':int(((cp<.02)&(hp<.02)).sum()),'soft_positive_mass':float(pr.sum()),'training_pools_exclude_outer_validation':True})
    print('future event',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2))
                                                                     
 from session12_event_replacement import assemble
 assemble(ROOT)
if __name__=='__main__':main()
