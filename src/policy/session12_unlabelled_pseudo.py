\
\
\
import os,json,time,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sequence_features import augment
import build_outcome_roles as BOR,build_relationship_evidence as BRE
from session8_data import hand_data,targets
from session6_priority_model import load_models as cats
from session7_model import load_models as events
CACHE=Path('artifacts/evidence_session12/unlabelled_pseudo');NATIVE=os.environ.get('PSEUDO_NATIVE')=='1';GATED=NATIVE or os.environ.get('PSEUDO_PAIR_GATE')=='1';ROOT=Path('artifacts/evidence_session12/unlabelled_native') if NATIVE else (Path('artifacts/evidence_session12/unlabelled_gated') if GATED else CACHE);C=pl.col
CONFIG={'registry_sample':'SHA256(pair_id) first byte even; no score-based selection','teacher':'Cat/HGB event heads outer f; no validation-pool gameplay added','positive_agreement':.9,'negative_agreement':.02,'pair_minimum_confident_hands':2,'family_routing':'highest confident-hand count, then total mean event mass','pseudo_weight_mass_fraction':.5,'soft_targets':'arithmetic Cat/HGB event average, duplicated fractional class labels','student':'unchanged 400-tree depth-five Cat event head and archived seed'}
def cache():
 cm,cols=cats('priority_ordered');hm,_=events('hist_eventblend');reg=pl.read_csv('data/evaluation_pairs.csv');reg=reg.filter(pl.Series([hashlib.sha256(p.encode()).digest()[0]%2==0 for p in reg['pair_id']]));players=reg.select('pair_id','player_1','player_2');folds=json.load(open('artifacts/policy/table_folds.json'));fams=['directed_transfer','soft_play','coordinated_isolation'];start=time.time();audits=[]
 for f in range(4):(ROOT/f'outer{f}').mkdir(exist_ok=True)
 with threadpool_limits(limits=2):
  for i,path in enumerate(sorted(Path('artifacts/detail_features').glob('T*.parquet'))):
   held=folds[path.stem];need=[f for f in range(4) if f!=held and not (ROOT/f'outer{f}'/path.name).exists()]
   if not need:continue
   d=pl.read_parquet(path).filter(C('phase')=='evaluation').join(players.select('pair_id'),on='pair_id').with_columns(((C('time')-.6)/.4).alias('relative_time'))
   if not len(d):continue
   h=pl.read_parquet(Path('artifacts/policy/hand_features')/path.name).filter(C('phase')=='evaluation').join(players.select('pair_id'),on='pair_id').sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')];h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')];h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']];n=len(d);d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id'],validate='1:1');assert len(d)==n;d,_=augment(d);query=d.select('pair_id','hand_id').join(players,on='pair_id');d=d.join(BOR.build(path.stem,query),on=['pair_id','hand_id']).join(BRE.build(path.stem,query),on=['pair_id','hand_id']).sort('pair_id','time','hand_id');x=d.select(cols).to_numpy();assert np.isfinite(x).all();groups=[];off=0
   for (pid,),g in d.group_by('pair_id',maintain_order=True):groups.append((pid,off,off+len(g)));off+=len(g)
   for f in need:
    cp=np.stack([np.column_stack([cm[b][f][k].predict_proba(x,thread_count=2)[:,1] for k in range(2)]) for b in fams],1);hp=np.stack([np.column_stack([hm[b][f][1][k].predict_proba(x)[:,1] for k in range(2)]) for b in fams],1);high=(cp>.9)&(hp>.9);low=(cp<.02)&(hp<.02);avg=.5*(cp+hp);indices=[];family=[];probs=[];masks=[];pairs=0
    for pid,lo,hi in groups:
     count=high[lo:hi].any(2).sum(0);mass=avg[lo:hi].sum((0,2));best=max(range(3),key=lambda j:(count[j],mass[j]));
     if count[best]<2:continue
     use=(high[lo:hi,best]|low[lo:hi,best]);keep=use.any(1);ix=np.arange(lo,hi)[keep];indices.extend(ix);family.extend([fams[best]]*len(ix));probs.append(avg[ix,best]);masks.append(use[keep]);pairs+=1
    base=d.select('pair_id','hand_id',*[C(c).cast(pl.Float32) for c in cols]);ix=np.array(indices,dtype=np.int64);out=base[ix];pr=np.concatenate(probs) if probs else np.zeros((0,2));el=np.concatenate(masks) if masks else np.zeros((0,2),bool);out=out.with_columns(pl.Series('behavior_family',family,dtype=pl.String),pl.lit(held).alias('pool_fold'),pl.lit(f).alias('teacher_fold'),pl.Series('pseudo_primary',pr[:,0]),pl.Series('pseudo_secondary',pr[:,1]),pl.Series('eligible_primary',el[:,0]),pl.Series('eligible_secondary',el[:,1]));out.write_parquet(ROOT/f'outer{f}'/path.name,compression='zstd');audit={'table_id':path.stem,'pool_fold':held,'teacher_fold':f,'registry_pairs_scanned':len(groups),'hands_scanned':len(d),'selected_pairs':pairs,'selected_hands':len(out),'positive_soft_mass':pr.sum(0).tolist()};(ROOT/f'outer{f}'/path.with_suffix('.json').name).write_text(json.dumps(audit,indent=2));audits.append(audit)
   if i%20==0:print('unlabelled scan',i,'seconds',round(time.time()-start,1),'recent selected pairs',sum(a['selected_pairs'] for a in audits[-3:]),flush=True)
 all_audit=[]
 for f in range(4):
  for path in (ROOT/f'outer{f}').glob('T*.json'):all_audit.append(json.load(open(path)))
 (ROOT/'cache_audit.json').write_text(json.dumps({'sampled_registry_pairs':len(reg),'tables':all_audit},indent=2))

def train():
 d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();fams=['directed_transfer','soft_play','coordinated_isolation'];pred=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  q=pl.read_parquet(list((CACHE/f'outer{f}').glob('T*.parquet')));
  if GATED:q=q.join(pl.read_parquet(f'artifacts/evidence_session12/pseudo_pair_gate/gate_fold{f}.parquet').filter(C('keep')).select('pair_id'),on='pair_id',validate='m:1')
  assert (q['pool_fold']!=f).all() and (q['teacher_fold']==f).all();assert q.select('pair_id','hand_id').n_unique()==len(q)
  for fam in fams:
   z=q.filter(C('behavior_family')==fam);ZX=z.select(cols).to_numpy();a,b,ea,eb,va=targets(d,f,fam)
   for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
    col='primary' if k==1 else 'secondary';use=z['eligible_'+col].to_numpy();px=ZX[use];pr=z['pseudo_'+col].to_numpy()[use];ix=np.flatnonzero(e);assert not np.any(e&va);weight=.5*len(ix)/max(1,len(px));tx=np.concatenate([X[ix],px,px]);ty=np.r_[y[ix].astype(int),np.zeros(len(px),int),np.ones(len(px),int)];w=np.r_[np.ones(len(ix)),weight*(1-pr),weight*pr];
    if NATIVE:tx=np.concatenate([X[ix],px]);ty=np.r_[y[ix].astype(float),pr];w=np.r_[np.ones(len(ix)),np.full(len(px),weight)];w=w/w.mean()
    m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='CrossEntropy' if NATIVE else 'Logloss',thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);path=ROOT/f'event{k}_{fam}_fold{f}.cbm'
    if path.exists():m.load_model(str(path))
    else:m.fit(tx,ty,sample_weight=w);m.save_model(str(path))
    pred[va,k-1]=m.predict_proba(X[va],thread_count=3)[:,1];audit.append({'fold':f,'family':fam,'event':k,'original_eligible':len(ix),'original_positive':int(y[ix].sum()),'pseudo_eligible':len(px),'pseudo_positive_soft_mass':float(pr.sum()),'pseudo_weight':weight,'pseudo_pairs':z['pair_id'].n_unique()})
   print('unlabelled student',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2));from session12_event_replacement import assemble;assemble(ROOT)
if __name__=='__main__':
 ROOT.mkdir(exist_ok=True);CONFIG['pair_gate']=GATED;CONFIG['native_soft_targets_mean_one_weights']=NATIVE;(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));
 if not GATED:cache()
 train()
