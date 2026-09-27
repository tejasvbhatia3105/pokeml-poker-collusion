\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('POKER_PARTITION_ROOT','artifacts/compact')
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
from session6_priority_model import load_models as cats
from session7_model import load_models as hist
from session8_data import hand_data,targets
ROOT=Path('artifacts/evidence_session12/transductive');C=pl.col
SCRATCH=Path('cache/hand_rows2')
CONFIG={'protocol':'transductive held-out development-pool gameplay; all candidate pairs scored without reading their true labels','pair_gate':.9,'minimum_confident_hands':2,'positive_agreement':.9,'negative_agreement':.02,'pseudo_mass_fraction':.5,'event_learner':'original 400-tree Cat; soft targets via weighted duplication','teachers':'fresh fixed 600-tree pair model and archived event heads excluding outer fold','no_oracle_selection':'candidate registry is all pair_features development rows, not public positives or evidence-hand lists'}
def cache():
 folds=json.load(open('artifacts/policy/table_folds.json'));pc=json.load(open('artifacts/policy/residual_columns.json'));cm,cols=cats('priority_ordered');hm,_=hist('hist_eventblend');pm=[];fams=['directed_transfer','soft_play','coordinated_isolation'];start=time.time()
 for f in range(4):
  m=CatBoostClassifier();m.load_model(f'artifacts/evidence_session12/pseudo_pair_gate/pair_fold{f}.cbm');pm.append(m);(ROOT/f'outer{f}').mkdir(exist_ok=True)
 with threadpool_limits(limits=2):
  for i,path in enumerate(sorted(Path('artifacts/policy/pair_features').glob('T*.parquet'))):
   f=folds[path.stem];out=ROOT/f'outer{f}'/path.name
   if out.exists():continue
   pairs=pl.read_parquet(path).filter(C('phase')=='development');probs=pm[f].predict_proba(pairs.select(pc).to_numpy(),thread_count=2);families=np.array(fams)[probs[:,1:].argmax(1)];pairs=pairs.with_columns(pl.Series('risk',1-probs[:,0]),pl.Series('behavior_family',families)).filter(C('risk')>=.9);players=pairs.select('pair_id','player_1','player_2');meta={'table_id':path.stem,'outer_fold':f,'all_candidate_pairs':len(probs),'pair_teacher_selected':len(pairs)}
   if not len(players):
                                                                          
    out.with_suffix('.json').write_text(json.dumps(meta|{'selected_pseudo_pairs':0,'selected_hands':0},indent=2));continue
   raw=basic(path.stem,players,players.head(0));d=detail(path.stem,base=raw,pairs=players).join(pairs.select('pair_id','behavior_family'),on='pair_id').with_columns((C('time')/.6).alias('relative_time'));h=pl.read_parquet(SCRATCH/path.name).select(list(pl.read_parquet_schema(Path('artifacts/policy/hand_features')/path.name))).filter(C('phase')=='development').drop('pair_id').join(players,on=['player_1','player_2']).sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')];h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')];h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']];n=len(d);d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id'],validate='1:1');assert len(d)==n;d,_=augment(d);query=d.select('pair_id','hand_id').join(players,on='pair_id');d=d.join(BOR.build(path.stem,query),on=['pair_id','hand_id']).join(BRE.build(path.stem,query),on=['pair_id','hand_id']);parts=[];selected=[]
   for fam in fams:
    z=d.filter(C('behavior_family')==fam).sort('pair_id','time','hand_id');x=z.select(cols).to_numpy()
    if not len(z):continue
    cp=np.column_stack([cm[fam][f][k].predict_proba(x,thread_count=2)[:,1] for k in range(2)]);hp=np.column_stack([hm[fam][f][1][k].predict_proba(x)[:,1] for k in range(2)]);high=(cp>.9)&(hp>.9);low=(cp<.02)&(hp<.02);p=.5*(cp+hp);z=z.with_columns(pl.Series('confident',high.any(1)),pl.Series('pseudo_primary',p[:,0]),pl.Series('pseudo_secondary',p[:,1]),pl.Series('eligible_primary',(high|low)[:,0]),pl.Series('eligible_secondary',(high|low)[:,1]));keep=z.group_by('pair_id').agg(C('confident').sum()).filter(C('confident')>=2).select('pair_id');z=z.join(keep,on='pair_id').filter(C('eligible_primary')|C('eligible_secondary'));selected.extend(keep['pair_id']);parts.append(z.select('pair_id','hand_id','behavior_family','pseudo_primary','pseudo_secondary','eligible_primary','eligible_secondary',*[C(c).cast(pl.Float32) for c in cols]).with_columns(pl.lit(f).alias('outer_fold')))
   result=pl.concat(parts);result.write_parquet(out,compression='zstd');out.with_suffix('.json').write_text(json.dumps(meta|{'selected_pseudo_pairs':len(selected),'selected_hands':len(result),'source':'unlabeled held-out gameplay; teacher excludes this outer fold'},indent=2))
   if i%40==0:print('transductive feature tables',i,'seconds',round(time.time()-start,1),flush=True)

def train():
 d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();pred=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  q=pl.read_parquet(list((ROOT/f'outer{f}').glob('T*.parquet')));assert (q['outer_fold']==f).all();assert q.select('pair_id','hand_id').n_unique()==len(q)
  for fam in ['directed_transfer','soft_play','coordinated_isolation']:
   z=q.filter(C('behavior_family')==fam);ZX=z.select(cols).to_numpy();a,b,ea,eb,va=targets(d,f,fam)
   for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
    assert not np.any(e&va);col='primary' if k==1 else 'secondary';use=z['eligible_'+col].to_numpy();px=ZX[use];pr=z['pseudo_'+col].to_numpy()[use];ix=np.flatnonzero(e);w0=.5*len(ix)/max(1,len(px));tx=np.r_[X[ix],px,px];ty=np.r_[y[ix].astype(int),np.zeros(len(px),int),np.ones(len(px),int)];w=np.r_[np.ones(len(ix)),w0*(1-pr),w0*pr];m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);path=ROOT/f'event{k}_{fam}_fold{f}.cbm'
    if path.exists():m.load_model(str(path))
    else:m.fit(tx,ty,sample_weight=w);m.save_model(str(path))
    pred[va,k-1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'family':fam,'kind':k,'original_labelled_rows':len(ix),'unlabelled_target_rows':len(px),'soft_positive_mass':float(pr.sum()),'real_validation_labels_excluded':True})
   print('transductive event',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2));from session12_event_replacement import assemble;assemble(ROOT)
if __name__=='__main__':
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));cache();train()
