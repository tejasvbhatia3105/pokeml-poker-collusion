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
from catboost import CatBoostClassifier
from session55_current_targets import state
from session8_data import hand_data,targets
from session35_fold_likelihood import labels
from session29_shared_fold_witness import target
from session39_bet_call import targets as secondary_targets
from session25_persistent_actor import actor_features
from session57_isolation_pressure import data as pressure_data,noisy_or
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session82_seed_stability');C=pl.col;OFFSETS=[0,10000,20000]
def fit(x,y,tr,seed,path,loss='Logloss'):
 m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function=loss,random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(path));return m
def main():
 ROOT.mkdir(exist_ok=True);full=hand_data();states=state();base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');parts={offset:[] for offset in OFFSETS};audit=[];start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'offsets':OFFSETS},indent=2))
 for family in ['directed_transfer','soft_play']:
  v=states[family];d=v['d'];a=v['a'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];direct=family=='directed_transfer';AX=actor_features(d) if direct else None;dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if direct else np.ones((len(d),2));pred={offset:np.zeros((len(d),2)) for offset in OFFSETS};fm=full['behavior_family'].to_numpy()==family
  for f in range(4):
   if direct:y,tr,va=labels(d,a,f);ys,es,donor=secondary_targets(d,f);selected=AX[np.arange(len(d)),np.maximum(donor,0)];sx=np.column_stack([v['hx'],selected])
   else:
    yy,e,_=target(d,f);y=yy[g];tr=e[g];va=fv[g]==f;_,y2,_,e2,_=targets(full,f,family);ys=y2[fm];es=e2[fm];sx=v['hx']
   for offset in OFFSETS:
    root=ROOT/f'seed{offset}';root.mkdir(exist_ok=True)
    for head in [0,1]:
     tx,ty,train=(v['x'],y,tr) if head==0 else (sx,ys,es);valid=va if head==0 else fv==f;assert not (train&valid).any();path=root/f'{family}_head{head+1}_fold{f}.cbm'
     if offset:m=fit(tx,ty,train,6311+11*f+head+offset,path)
     else:
      source=f'artifacts/evidence_session50_matchup/current/{family}/event1_fold{f}.cbm' if head==0 else (f'artifacts/evidence_session25_persistent_actor/oriented/event2_directed_transfer_fold{f}.cbm' if direct else f'artifacts/evidence_session6/priority_ordered_event2_soft_play_fold{f}.cbm');m=CatBoostClassifier();m.load_model(source)
     if head==0:
      p=np.zeros((len(d),2));p[g[valid],actor[valid]]=m.predict_proba(tx[valid],thread_count=2)[:,1];pred[offset][fv==f,0]=(p*dw).sum(1)[fv==f]
     elif direct:
      p=np.column_stack([m.predict_proba(np.column_stack([v['hx'][valid],AX[valid,r]]),thread_count=2)[:,1] for r in [0,1]]);pred[offset][valid,1]=(p*dw[valid]).sum(1)
     else:pred[offset][valid,1]=m.predict_proba(tx[valid],thread_count=2)[:,1]
     audit.append({'family':family,'fold':f,'offset':offset,'head':head+1,'training_rows':int(train.sum()),'positives':int(ty[train].sum()),'validation_overlap':0})
   print('seed stability',family,f,round(time.time()-start,1),flush=True)
  for offset in OFFSETS:parts[offset].append(d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',pred[offset][:,0]),pl.Series('new_secondary',pred[offset][:,1])))
 _,d,a,ac=pressure_data();g=a['row'].to_numpy();cfg=json.load(open('artifacts/evidence_session59_pressure_equity/config.json'));x=np.column_stack([a.select(ac).to_numpy(),d.select(cfg['hand_columns']).to_numpy()[g],np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x']]);fv=d['fold'].to_numpy();cnt=np.bincount(g,minlength=len(d));fm=full['behavior_family'].to_numpy()=='coordinated_isolation';pred={offset:np.zeros((len(d),2)) for offset in OFFSETS}
 for f in range(4):
  p1,p2,e1,e2,_=targets(full,f,'coordinated_isolation');ys=np.column_stack([p1,p2])[fm];es=np.column_stack([e1,e2])[fm];va=fv[g]==f
  for offset in OFFSETS:
   root=ROOT/f'seed{offset}'
   for head in [0,1]:
    y=ys[:,head];tr=es[g,head];assert not (tr&va).any();gt=g[tr];yt=y[gt];resp=yt/cnt[gt]
    if offset:
     for em in range(3):
      yy=np.zeros(len(a));yy[tr]=resp;m=fit(x,yy,tr,6311+11*f+head+offset,root/f'coordinated_isolation_head{head+1}_fold{f}_em{em}.cbm',loss='CrossEntropy');p=m.predict_proba(x[tr],thread_count=2)[:,1];hp=noisy_or(p,gt,len(d));resp=np.where(yt,p/np.maximum(hp[gt],1e-8),0).clip(0,1)
    else:m=CatBoostClassifier();m.load_model(f'artifacts/evidence_session59_pressure_equity/event{head+1}_fold{f}_em2.cbm')
    hp=noisy_or(m.predict_proba(x[va],thread_count=2)[:,1],g[va],len(d));pred[offset][fv==f,head]=hp[fv==f];audit.append({'family':'coordinated_isolation','fold':f,'offset':offset,'head':head+1,'training_actions':int(tr.sum()),'positive_bags':int(y.sum()),'validation_overlap':0})
  print('seed stability isolation',f,round(time.time()-start,1),flush=True)
 for offset in OFFSETS:
  parts[offset].append(d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',pred[offset][:,0]),pl.Series('new_secondary',pred[offset][:,1])));q=base.join(pl.concat(parts[offset]),on=['pair_id','hand_id'],validate='1:1')
  if offset==0:
   for a,b in [('new_primary','bg_primary'),('new_secondary','bg_secondary')]:np.testing.assert_array_equal(q[a].to_numpy(),q[b].to_numpy())
  out=q.with_columns(C('new_primary').alias('bg_primary'),C('new_secondary').alias('bg_secondary')).drop('new_primary','new_secondary');out.write_parquet(ROOT/f'seed{offset}'/'event_oof.parquet');assemble(ROOT/f'seed{offset}')
 (ROOT/'audit.json').write_text(json.dumps({'method':__doc__,'original_control_exact':True,'fits':audit},indent=2))
if __name__=='__main__':main()
