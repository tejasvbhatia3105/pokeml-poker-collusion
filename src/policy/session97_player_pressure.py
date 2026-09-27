\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session96_player_context_evidence as s
from session57_isolation_pressure import data,noisy_or
from session8_data import targets
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session97_player_pressure');C=pl.col;KINDS=s.KINDS
CONFIG={'method':__doc__,'arms':KINDS,'context':s.CONFIG,'fields':['ordinary_p0','ordinary_p1','ordinary_p2','ordinary_p3','raise_surprisal','size_residual','absolute_size_residual'],'schedule':'59 original3 EM Cat400 D5 lr.035 L2=8 seed6311+11fold+head','no_family_or_weight_selection':True,'pretraining_limitation':s.CONFIG['policy_representation']}
def policy_features(q,h,native,ms):
 action,size,meta=ms;p=s.p;x=q.select(p.s.PC).to_numpy();hx=h.select(p.s.PC).to_numpy();pr={key:m.predict_proba(x,thread_count=2) for key,m in action.items() if native in key};hr={key:action[key].predict_proba(hx,thread_count=2) for key in pr};sz={key:size[key].predict(x,thread_count=2) for key in pr};out={}
 for f in range(4):
  key=tuple(sorted([native,f]));pp=p.legal(sum(pr.values())/3,x) if native==f else p.legal(pr[key],x);hp=p.legal(sum(hr.values())/3,hx) if native==f else p.legal(hr[key],hx);sizepred=sum(sz.values())/3 if native==f else sz[key];prior=np.log(pp.clip(1e-7)).astype(np.float32);base=np.column_stack([x,prior]);ex=p.fields(q,h,pp,hp);res=q['log_bet_ratio'].to_numpy()-sizepred
  for kind in KINDS:
   xx=np.column_stack([base,ex]) if kind=='context' else base;prob=p.predict(meta[f,kind],xx,prior);out[f,kind]=np.column_stack([prob,-np.log(prob[:,3].clip(1e-7)),res,abs(res)]).astype(np.float32)
 return out
def prepare():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));ms=s.models();_,d,a,ac=data();a=a.with_row_index('action_row');keys=a.select('action_row','pair_id','hand_id',C('action_no').cast(pl.Int64)).with_columns(C('action_row').alias('query_row'),pl.lit(0).alias('role'));values={(f,k):np.zeros((len(a),7),np.float32) for f in range(4) for k in KINDS};raw=[];audit=[];labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');start=time.time()
 for (table,),group in d.group_by('table_id'):
  src=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');local=keys.join(group.select('pair_id').unique(),on='pair_id',how='semi');local=local.join(src.select('hand_id',C('action_no').cast(pl.Int64),C('player_id').alias('expected_player')),on=['hand_id','action_no'],validate='m:1');z=local.join(labs,on='pair_id',validate='m:1');assert ((z['expected_player']==z['player_1'])|(z['expected_player']==z['player_2'])).all()
  for (pid,),query in local.group_by('pair_id'):
   pair=group.filter(C('pair_id')==pid);q,h=s.pair_input(src,query,set(pair['hand_id']));assert (q['action_class']==3).all();pred=policy_features(q,h,int(pair['fold'][0]),ms);ix=q['action_row'].to_numpy();raw.append(q)
   for fk,z in pred.items():values[fk][ix]=z
   audit.append({'pair_id':pid,'table_id':table,'query_actions':len(q),'context_actions':len(h),'excluded_shared_hands':len(pair),'query_context_overlap':0})
 z=pl.concat(raw).sort('query_row');np.testing.assert_array_equal(z['query_row'].to_numpy(),np.arange(len(a)));z.write_parquet(ROOT/'queries.parquet')
 for (f,k),x in values.items():
  root=ROOT/k;root.mkdir(exist_ok=True);assert (x[:,:4].sum(1)>.99).all();np.savez_compressed(root/f'extra_fold{f}.npz',x=x)
 (ROOT/'input_audit.json').write_text(json.dumps(audit,indent=2));print('pressure context prepared',len(a),round(time.time()-start,1),flush=True)
def train():
 full,d,a,ac=data();g=a['row'].to_numpy();cnt=np.bincount(g,minlength=len(d));fv=d['fold'].to_numpy();hc=json.load(open('artifacts/evidence_session59_pressure_equity/config.json'))['hand_columns'];base_x=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x']]);fm=full['behavior_family'].to_numpy()=='coordinated_isolation';pred={k:np.zeros((len(d),2)) for k in KINDS};audit=[];start=time.time()
 for f in range(4):
  p1,p2,e1,e2,_=targets(full,f,'coordinated_isolation');ys=np.column_stack([p1,p2])[fm];es=np.column_stack([e1,e2])[fm];va=fv[g]==f
  for k in KINDS:
   root=ROOT/k;x=np.column_stack([base_x,np.load(root/f'extra_fold{f}.npz')['x']])
   for head in range(2):
    y=ys[:,head];tr=es[g,head];assert not(tr&va).any();gt=g[tr];yt=y[gt];resp=yt/cnt[gt]
    for em in range(3):
     m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='CrossEntropy',random_seed=6311+11*f+head,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],resp);m.save_model(str(root/f'head{head+1}_fold{f}_em{em}.cbm'));p=m.predict_proba(x[tr],thread_count=2)[:,1];hp=noisy_or(p,gt,len(d));resp=np.where(yt,p/np.maximum(hp[gt],1e-8),0).clip(0,1)
    hp=noisy_or(m.predict_proba(x[va],thread_count=2)[:,1],g[va],len(d));pred[k][fv==f,head]=hp[fv==f];audit.append({'fold':f,'kind':k,'head':head+1,'training_actions':int(tr.sum()),'validation_overlap':0})
   print('pressure context',f,k,round(time.time()-start,1),flush=True)
 for k in KINDS:
  root=ROOT/k;base=pl.read_parquet(s.ROOT/k/'event_oof.parquet');z=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',pred[k][:,0]),pl.Series('new_secondary',pred[k][:,1]));base.join(z,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary'),pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_primary','new_secondary').write_parquet(root/'event_oof.parquet');assemble(root)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 else:train()
