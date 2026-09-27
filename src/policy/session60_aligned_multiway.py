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
import session59_pressure_equity as precision
from session50_matchup import load,ROOT as MATCHUP,labels,target,OLD,C,assemble
ROOT=Path('artifacts/evidence_session60_aligned_multiway')
def design(family):
 d,a,x,cols=load(family);current=np.load(MATCHUP/family/'features.npz')['x'][:,[3,6]];return d,a,np.column_stack([x,current])
def main():
 ROOT.mkdir(exist_ok=True);base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');parts=[];audit=[];start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'extra_columns':precision.COLS,'schedule':'R32 Cat400 D5 lr.035 L2=8 seeds6311+11fold','baseline':'59 precise pressure, .795147103 local MAP','selection':'both direct/soft primary heads updated, isolation and all secondary unchanged'},indent=2))
 for family in ['directed_transfer','soft_play']:
  folder=ROOT/family;folder.mkdir(exist_ok=True);d,a,x=design(family);precision.ROOT=folder;ex,rawaudit=precision.compute(d,a);np.savez_compressed(folder/'features.npz',x=ex);(folder/'feature_audit.json').write_text(json.dumps(rawaudit,indent=2));xx=np.column_stack([x,ex]);g=a['row'].to_numpy();r=a['actor'].to_numpy();pp=np.zeros((len(d),2))
  for f in range(4):
   if family=='directed_transfer':y,tr,va=labels(d,a,f)
   else:y0,e,_=target(d,f);y=y0[g];tr=e[g];va=d['fold'].to_numpy()[g]==f
   assert not(tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(xx[tr],y[tr]);m.save_model(str(folder/f'primary_fold{f}.cbm'));pp[g[va],r[va]]=m.predict_proba(xx[va],thread_count=2)[:,1];audit.append({'family':family,'fold':f,'training_actions':int(tr.sum()),'positive_actions':int(y[tr].sum()),'validation_overlap':0});print('aligned precise',family,f,round(time.time()-start,1),flush=True)
  d.select('pair_id','hand_id').with_columns(pl.Series('actor0_primary',pp[:,0]),pl.Series('actor1_primary',pp[:,1])).write_parquet(folder/'conditional_oof.parquet');dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if family=='directed_transfer' else np.ones_like(pp);parts.append(d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',(pp*dw).sum(1))))
 base.join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary')).drop('new_primary').write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':main()
