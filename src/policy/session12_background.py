import os,json,time,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sequence_features import augment
import build_outcome_roles as BOR,build_relationship_evidence as BRE
from session8_data import hand_data,targets,reference
from session6_priority import inclusion
from session8_count_conditioning import conditioned
from session11_conditional_family import Model,features
ROOT=Path('artifacts/evidence_session12/background');C=pl.col
CONFIG={'negative_hands_per_pair':16,'negative_weight_mass_fraction':.25,'iterations':400,'depth':5,'learning_rate':.035,'l2_leaf_reg':8,'selection':'deterministic uniform sample after full-history feature construction; only confirmed_non_target; unknown excluded','positive_targets':'unchanged archived ordered subtype and censoring training targets','seeds':'6310+11*fold+event_kind, exact archived Cat event seed'}
def background():
 path=ROOT/'negative_features.parquet'
 if path.exists():return pl.read_parquet(path)
 cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];labs=pl.read_csv('data/development_labels.csv').filter(C('label')==0).select('pair_id','player_1','player_2');folds=json.load(open('artifacts/policy/table_folds.json'));parts=[];start=time.time()
 for i,path0 in enumerate(sorted(Path('artifacts/detail_features').glob('T*.parquet'))):
  d=pl.read_parquet(path0).filter(C('phase')=='development').join(labs.select('pair_id'),on='pair_id').with_columns((C('time')/.6).alias('relative_time'))
  if not len(d):continue
  h=pl.read_parquet(Path('artifacts/policy/hand_features')/path0.name).filter(C('phase')=='development').join(labs.select('pair_id'),on='pair_id').sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')];h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')];h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']];d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']);d,_=augment(d);query=d.select('pair_id','hand_id').join(labs,on='pair_id');d=d.join(BOR.build(path0.stem,query),on=['pair_id','hand_id']).join(BRE.build(path0.stem,query),on=['pair_id','hand_id']);take=[]
  for (pid,),g in d.group_by('pair_id'):
   g=g.sort('time','hand_id');seed=int(hashlib.sha256(pid.encode()).hexdigest()[:8],16);ix=np.random.default_rng(seed).choice(len(g),min(CONFIG['negative_hands_per_pair'],len(g)),replace=False);take.append(g[ix].select('pair_id','hand_id',*[C(c).cast(pl.Float32) for c in cols]).with_columns(pl.lit(folds[path0.stem]).alias('fold')))
  parts.extend(take)
  if i%80==0:print('background features',i,round(time.time()-start,1),flush=True)
 neg=pl.concat(parts);assert neg['pair_id'].n_unique()==len(labs);assert np.isfinite(neg.select(cols).to_numpy()).all();neg.write_parquet(path,compression='zstd');return neg

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));neg=background();d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();NX=neg.select(cols).to_numpy();nf=neg['fold'].to_numpy();r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet');pred=np.zeros((len(d),2));audit=[];start=time.time()
 with threadpool_limits(limits=3):
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    a,b,ea,eb,va=targets(d,f,fam)
    for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
     ni=np.flatnonzero(nf!=f);ix=np.flatnonzero(e);assert not np.any(e&va);trainX=np.r_[X[ix],NX[ni]];trainY=np.r_[y[ix].astype(int),np.zeros(len(ni),int)];w=np.r_[np.ones(len(ix)),np.full(len(ni),CONFIG['negative_weight_mass_fraction']*len(ix)/len(ni))];m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=3,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);modelpath=ROOT/f'event{k}_{fam}_fold{f}.cbm'
     if modelpath.exists():m.load_model(str(modelpath))
     else:m.fit(trainX,trainY,sample_weight=w)
     pred[va,k-1]=m.predict_proba(X[va],thread_count=3)[:,1];m.save_model(str(ROOT/f'event{k}_{fam}_fold{f}.cbm'));audit.append({'fold':f,'family':fam,'kind':k,'positive_pair_hands':len(ix),'positive_events':int(y[ix].sum()),'confirmed_negative_hands':len(ni),'negative_weight_per_hand':float(w[-1]),'heldout_pool_negative_hands_excluded':int((nf==f).sum())})
    print('background event',f,fam,round(time.time()-start,1),flush=True)
 q=d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1]));q.write_parquet(ROOT/'event_oof.parquet');parts=[]
 for f in range(4):
  raw=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time');z=d.filter(C('fold')==f).join(raw,on=['pair_id','hand_id'],validate='1:1').join(q.select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1').join(r30.select('pair_id','hand_id','conditional_family'),on=['pair_id','hand_id'],validate='1:1');models=[]
  for seed in [1010,2020]:
   st=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(st['state_dict']);m.eval();models.append((st,m))
  for (pid,),g in z.group_by('pair_id'):
   g=g.sort('time','hand_id');ca=g.select('bg_primary','bg_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];ci=inclusion(ca[:,0],ca[:,1]);base=g['conditional_family'].to_numpy();replace=base+.25*(ci-g['cat_inclusion'].to_numpy());hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];jp=.5*(ca+hp);prior=np.log(np.maximum(jp,1e-6))-np.log(np.maximum(1-jp.sum(1),1e-6))[:,None];x,_=features(g);incs=[]
   with torch.no_grad():
    for st,m in models:
     delta=m(torch.tensor(np.clip((x-st['mu'])/st['sd'],-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];logits=torch.tensor(prior,dtype=torch.float32)+delta;p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:1]),logits],1),1).numpy();incs.append(conditioned(p[:,1:],st['minimums'][g['behavior_family'][0]]))
   allreplace=.25*g['base'].to_numpy()+.25*ci+.5*np.mean(incs,0);parts.append(g.select('pair_id','hand_id').with_columns(pl.Series('cat_only',replace),pl.Series('cat_and_joint',allreplace)))
 pl.concat(parts).write_parquet(ROOT/'background_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
