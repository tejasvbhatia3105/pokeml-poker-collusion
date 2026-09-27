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
from session8_data import hand_data
from session55_current_targets import direct
from session25_persistent_actor import actor_features
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session81_conditional_secondary');C=pl.col
def primary_by_actor(d):
 from session50_matchup import load
 dd,a,x,_=load('directed_transfer');assert dd.select('pair_id','hand_id').equals(d.select('pair_id','hand_id'));x=np.column_stack([x,np.load('artifacts/evidence_session50_matchup/directed_transfer/features.npz')['x'][:,[3,6]]]);g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=d['fold'].to_numpy();out=np.zeros((len(d),2))
 for f in range(4):
  m=CatBoostClassifier();m.load_model(f'artifacts/evidence_session50_matchup/current/directed_transfer/event1_fold{f}.cbm');va=fv[g]==f;out[g[va],actor[va]]=m.predict_proba(x[va],thread_count=2)[:,1]
 return out
def rebuild_output():
 full=hand_data();d=full.filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();cond=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'conditional_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left').select('actor0_conditional','actor1_conditional').to_numpy();primary=primary_by_actor(d);base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');old=d.select('pair_id','hand_id').join(base,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');np.testing.assert_array_equal((primary*dw).sum(1),old['bg_primary'].to_numpy());secondary=((1-primary)*cond*dw).sum(1);assert (secondary+old['bg_primary'].to_numpy()<=1+1e-12).all();q=d.select('pair_id','hand_id').with_columns(pl.Series('replacement',secondary));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('replacement','bg_secondary').alias('bg_secondary')).drop('replacement').write_parquet(ROOT/'event_oof.parquet');assemble(ROOT)
def main():
 if os.environ.get('CONDITIONAL_REBUILD_ONLY')=='1':rebuild_output();return
 ROOT.mkdir(exist_ok=True);full=hand_data();d=full.filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));hx=d.select(cfg['event']).to_numpy();xt=d.select(cfg['type']).to_numpy();ax=actor_features(d);fv=d['fold'].to_numpy();dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');aligned=d.select('pair_id','hand_id').join(base,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');pred=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  typ=CatBoostClassifier();typ.load_model(f'artifacts/evidence_session6/priority_ordered_type_directed_transfer_fold{f}.cbm');tp=typ.predict_proba(xt,thread_count=2)[:,1];y,e,donor,_=direct(d,fv!=f,tp);tr=e[:,1]&~y[:,0];va=fv==f;assert not (tr&va).any();assert not (y[:,0]&y[:,1]).any();assert y[tr,1].sum()==y[:,1].sum();X=np.column_stack([hx[tr],ax[np.flatnonzero(tr),donor[tr]]]);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6312+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(X,y[tr,1]);m.save_model(str(ROOT/f'secondary_fold{f}.cbm'))
  for r in [0,1]:pred[va,r]=m.predict_proba(np.column_stack([hx[va],ax[va,r]]),thread_count=2)[:,1]
  audit.append({'fold':f,'secondary_training_rows':int(tr.sum()),'removed_primary_negatives':int((e[:,1]&y[:,0]).sum()),'positive_training_rows':int(y[tr,1].sum()),'validation_overlap':0});print('conditional secondary',f,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id').with_columns(pl.Series('actor0_conditional',pred[:,0]),pl.Series('actor1_conditional',pred[:,1])).write_parquet(ROOT/'conditional_oof.parquet');(ROOT/'audit.json').write_text(json.dumps({'method':__doc__,'fits':audit},indent=2));rebuild_output()
if __name__=='__main__':main()
