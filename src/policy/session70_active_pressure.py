\
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
import session69_pressure_relabel as relabel
from session57_isolation_pressure import data,BASE,noisy_or,assemble,C
from session67_isolation_types import design
from session59_pressure_equity import ROOT as PRECISE
ROOT=Path('artifacts/evidence_session70_active_pressure')
def main():
 ROOT.mkdir(exist_ok=True);full,d,a,ac=data();relabel.TYPE_X,tc=design(d,a,ac);maxactive=relabel.TYPE_X[:,tc.index('players_active_max')];hc=json.load(open(PRECISE/'config.json'))['hand_columns'];g=a['row'].to_numpy();ex=np.load(PRECISE/'features.npz')['x'];x=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],ex]);fv=d['fold'].to_numpy();sub=d['subtype'].to_numpy();mask=full['behavior_family'].to_numpy()=='coordinated_isolation';pp=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  targets=relabel.targets(full,f,'coordinated_isolation');ys=np.column_stack(targets[:2])[mask];es=np.column_stack(targets[2:4])[mask]
  for k in range(2):
   known=(fv!=f)&(sub==k+1);levels=np.unique(maxactive[known]);assert len(levels)==1;level=int(levels[0]);support=a['players_active'].to_numpy()==level;count=np.bincount(g[support],minlength=len(d));y=ys[:,k];tr=es[g,k]&support;va=(fv[g]==f)&support;assert not (tr&va).any();assert not (y&(count==0)).any();gt=g[tr];yt=y[gt];resp=yt/count[gt];bags=np.flatnonzero(es[:,k]&(count>0));trace=[]
   for step in range(3):
    m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='CrossEntropy',random_seed=6311+11*f+k,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],resp);m.save_model(str(ROOT/f'event{k+1}_fold{f}_em{step}.cbm'));p=m.predict_proba(x[tr],thread_count=2)[:,1];hp=noisy_or(p,gt,len(d));resp=np.where(yt,p/np.maximum(hp[gt],1e-8),0).clip(0,1);v=hp[bags].clip(1e-8,1-1e-8);trace.append(float(-(y[bags]*np.log(v)+(1-y[bags])*np.log1p(-v)).mean()))
   hp=noisy_or(m.predict_proba(x[va],thread_count=2)[:,1],g[va],len(d));pp[fv==f,k]=hp[fv==f];audit.append({'fold':f,'head':k+1,'active_level':level,'known_training_type_examples':int(known.sum()),'training_actions':int(tr.sum()),'positive_training_hands':int(y.sum()),'heldout_overlap':0,'bag_nll':trace});print('active pressure',f,k+1,round(time.time()-start,1),flush=True)
 base=pl.read_parquet(BASE/'event_oof.parquet');q=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',pp[:,0]),pl.Series('new_secondary',pp[:,1]));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary'),pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_primary','new_secondary').write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'action_columns':ac,'hand_columns':hc,'extra_columns':relabel.COLS},indent=2));assemble(ROOT)
if __name__=='__main__':main()
