\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier,Pool
from scipy.special import expit
from scipy.optimize import brentq
from session8_data import hand_data
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session35_fold_likelihood');OLD=Path('artifacts/evidence_session25_persistent_actor');EXACT=Path('artifacts/evidence_session26_exact_fold')

def data():
 d=hand_data().filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');a=pl.read_parquet(EXACT/'fold_actions.parquet').with_row_index('action_row');cfg=json.load(open(EXACT/'columns.json'));g=a['row'].to_numpy();x=np.column_stack([a.select(cfg['action']).to_numpy(),d.select(cfg['hand']).to_numpy()[g]]);pc=json.load(open('artifacts/policy/feature_columns.json'));parts=[]
 for (table,),q in d.group_by('table_id'):
  z=a.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');p=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').select('hand_id',*pc).with_columns(C('action_no').cast(pl.Int64));z=z.select('action_row','hand_id',C('action_no').cast(pl.Int64)).join(p,on=['hand_id','action_no'],validate='m:1');parts.append(z)
 p=pl.concat(parts).sort('action_row');assert np.array_equal(p['action_row'],np.arange(len(a)));px=p.select(pc).to_numpy();shared=[c for c in pc if c in cfg['action']];assert np.max(abs(p.select(shared).to_numpy()-a.select(shared).to_numpy()))==0
 return d,a,x,px,cfg,pc

def labels(d,a,f):
 by={r['pair_id']:r for r in json.load(open(OLD/f'assignments_fold{f}.json'))};y=np.zeros(len(d),bool);e=np.zeros(len(d),bool);donor=np.full(len(d),-1);time=d['time'].to_numpy()
 for (pid,),q in d.group_by('pair_id'):
  if pid not in by:continue
  assert q['fold'][0]!=f;ix=q['row'].to_numpy();pos=q.filter(C('evidence')==1).sort('evidence_rank')['row'].to_numpy();k=by[pid]['cut'];donor[ix]=by[pid]['donor'];e[ix]=True;y[pos[:k]]=True
  if k==5:e[ix[time[ix]>time[pos[-1]]]]=False
 g=a['row'].to_numpy();tr=e[g]&(a['actor'].to_numpy()==donor[g]);va=d['fold'].to_numpy()[g]==f;assert not (tr&va).any();assert y.sum()==y[g[tr]].sum();return y[g],tr,va

def ordinary(px,f):
 m=CatBoostClassifier();m.load_model(f'artifacts/policy/action_fold{f}.cbm');p=m.predict_proba(px,thread_count=2);p[:,1]=0;p/=p.sum(1)[:,None];return p

def main():
 ROOT.mkdir(exist_ok=True);d,a,x,px,cfg,pc=data();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'action_columns':cfg['action'],'hand_columns':cfg['hand'],'policy_columns':pc,'arms':['normal_features','surprisal_offset'],'schedule':'Cat400 depth5 lr.035 L2 8 original primary seed; same hard training assignments as26','ordinary_policy':'action_fold outer f on train and validation; no outer-f tables in its fitting; train predictions in-sample, validation held out','caveat':'normal policy is estimated from gameplay, can include anomalous actions; repeated-label model selection'},indent=2));g=a['row'].to_numpy();actor=a['actor'].to_numpy();dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();pred={k:np.zeros((len(d),2)) for k in ['normal_features','surprisal_offset']};audit=[];start=time.time()
 for f in range(4):
  p=ordinary(px,f);surprise=-np.log(p[:,0].clip(1e-6,1));xx=np.column_stack([x,p,surprise]);y,tr,va=labels(d,a,f);intercept=brentq(lambda b:expit(b+surprise[tr]).mean()-y[tr].mean(),-40,40);offset=surprise+intercept
  for kind in pred:
   root=ROOT/kind;root.mkdir(exist_ok=True);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6311+11*f,verbose=False,allow_writing_files=False);train=Pool(xx[tr],y[tr],baseline=offset[tr] if kind=='surprisal_offset' else None);m.fit(train);m.save_model(str(root/f'primary_fold{f}.cbm'));raw=m.predict(xx[va],prediction_type='RawFormulaVal',thread_count=2);pp=expit(raw+(offset[va] if kind=='surprisal_offset' else 0));pred[kind][g[va],actor[va]]=pp;audit.append({'fold':f,'kind':kind,'training_actions':int(tr.sum()),'positives':int(y[tr].sum()),'validation_actions':int(va.sum()),'validation_overlap':0,'intercept':float(intercept),'initial_mean_error':float(abs(expit(offset[tr]).mean()-y[tr].mean()))});print('fold likelihood',f,kind,round(time.time()-start,1),flush=True)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));base=pl.read_parquet(OLD/'oriented/event_oof.parquet')
 for kind,pp in pred.items():
  root=ROOT/kind;d.select('pair_id','hand_id').with_columns(pl.Series('actor0_primary',pp[:,0]),pl.Series('actor1_primary',pp[:,1])).write_parquet(root/'conditional_primary.parquet');q=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',(pp*dw).sum(1)));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary')).drop('new_primary').write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
