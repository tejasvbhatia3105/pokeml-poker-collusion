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
from session8_data import hand_data,targets
from session27_donor_call_witness import noisy_or
from session12_event_replacement import assemble
C=pl.col
ROOT=Path('artifacts/evidence_session57_isolation_pressure')
BASE=Path('artifacts/evidence_session50_matchup/current')

def data():
 full=hand_data();d=full.filter(C('behavior_family')=='coordinated_isolation').drop('row').with_row_index('row')
 a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet').filter((C('action_class')==3)&~C('facing_partner')&C('mw_alive')&(C('players_active')>=3))
 ac=[c for c in a.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family','time']]
 a=a.select('pair_id','hand_id',*ac).join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],how='inner',validate='m:1').sort('row','street_no','action_no')
 assert a.select('pair_id','hand_id','street_no','action_no').n_unique()==len(a)
 return full,d,a,ac

def main():
 ROOT.mkdir(exist_ok=True);full,d,a,ac=data();g=a['row'].to_numpy();cnt=np.bincount(g,minlength=len(d));support=cnt>0;truth=d['evidence'].to_numpy()==1
 support_audit={'hands':len(d),'actions':len(a),'supported_hands':int(support.sum()),'truth_hands':int(truth.sum()),'unsupported_truth_hands':int((truth&~support).sum()),'single_action_truth_hands':int((truth&(cnt==1)).sum()),'positive_action_counts':np.unique(cnt[truth],return_counts=True)[0].tolist(),'positive_action_frequencies':np.unique(cnt[truth],return_counts=True)[1].tolist()}
 (ROOT/'support_audit.json').write_text(json.dumps(support_audit,indent=2));print(json.dumps(support_audit),flush=True);assert not (truth&~support).any()
 hc=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];ax=a.select(ac).to_numpy();hx=d.select(hc).to_numpy();xs={'action':ax,'action_hand':np.column_stack([ax,hx[g]])};fv=d['fold'].to_numpy();mask=full['behavior_family'].to_numpy()=='coordinated_isolation'
 (ROOT/'config.json').write_text(json.dumps({'method':__doc__,'action_columns':ac,'hand_columns':hc,'support':'raise, not facing partner, partner alive, at least three active players','targets':'original session8 isolation targets and censoring; no heldout assignments','models':'3 EM rounds Cat400 D5 lr.035 L2=8 CrossEntropy, original head seeds6311+11fold+head','aggregation':'independent noisy-OR over supported outward raises','initial_responsibility':'positive hand mass divided equally among eligible actions','arms':['support_only',*xs]},indent=2));a.write_parquet(ROOT/'pressure_actions.parquet');base=pl.read_parquet(BASE/'event_oof.parquet');pp={k:np.zeros((len(d),2)) for k in xs};audit=[];start=time.time()
 for f in range(4):
  p1,p2,e1,e2,_=targets(full,f,'coordinated_isolation');ys=np.column_stack([p1,p2])[mask];es=np.column_stack([e1,e2])[mask];va=fv[g]==f
  for k in range(2):
   y=ys[:,k];eligible=es[:,k];tr=eligible[g];assert not(tr&va).any();assert int(y[eligible].sum())==int(y[eligible&support].sum());gt=g[tr];yt=y[gt];bags=np.flatnonzero(eligible&support)
   for kind,x in xs.items():
    root=ROOT/kind;root.mkdir(exist_ok=True);resp=yt/cnt[gt];trace=[]
    for step in range(3):
     m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='CrossEntropy',random_seed=6311+11*f+k,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],resp);path=root/f'event{k+1}_fold{f}_em{step}.cbm';m.save_model(str(path));reloaded=CatBoostClassifier();reloaded.load_model(str(path));p=m.predict_proba(x[tr],thread_count=2)[:,1];assert np.max(np.abs(p-reloaded.predict_proba(x[tr],thread_count=2)[:,1]))==0
     hp=noisy_or(p,gt,len(d));resp=np.where(yt,p/np.maximum(hp[gt],1e-8),0).clip(0,1);v=hp[bags].clip(1e-8,1-1e-8);trace.append(float(-(y[bags]*np.log(v)+(1-y[bags])*np.log1p(-v)).mean()))
    pred=noisy_or(m.predict_proba(x[va],thread_count=2)[:,1],g[va],len(d));pp[kind][fv==f,k]=pred[fv==f];audit.append({'fold':f,'head':k+1,'kind':kind,'training_actions':int(tr.sum()),'training_supported_hands':len(bags),'positive_hands':int(y[bags].sum()),'validation_overlap':0,'training_bag_nll':trace,'saved_model_reload_error':0});print('pressure',f,k+1,kind,round(time.time()-start,1),flush=True)
 for kind in ['support_only',*xs]:
  root=ROOT/kind;root.mkdir(exist_ok=True)
  if kind=='support_only':
   z=d.select('pair_id','hand_id').join(base,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');new=z.select('bg_primary','bg_secondary').to_numpy()*support[:,None]
  else:new=pp[kind]
  q=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',new[:,0]),pl.Series('new_secondary',new[:,1]));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary'),pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_primary','new_secondary').write_parquet(root/'event_oof.parquet');assemble(root)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
