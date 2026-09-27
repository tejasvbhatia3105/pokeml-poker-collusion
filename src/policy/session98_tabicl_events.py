\
\
\
\
\
\
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,time,gc,hashlib,importlib.metadata
from pathlib import Path
import numpy as np,polars as pl,torch
from catboost import CatBoostClassifier
from tabicl import TabICLClassifier
from session55_current_targets import state
from session35_fold_likelihood import labels
from session29_shared_fold_witness import target
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session98_tabicl');C=pl.col
CHECKPOINT=ROOT/'tabicl-classifier-v2-20260212.ckpt';SHA='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
CONFIG={'method':__doc__,'package':'tabicl2.2.0','checkpoint':CHECKPOINT.name,'checkpoint_sha256':SHA,'features':128,'feature_selection':'top original outer-Cat importances, ties by original feature index; retain original input order','TabICL':{'n_estimators':4,'batch_size':1,'device':'mps','n_jobs':2,'use_amp':False,'offload_mode':'cpu','safety_factor':.02,'random_state':'9800+fold','kv_cache':False},'Cat_control':'400 D5 lr.035 L2=8 seed6311+11fold','training_labels':'unchanged R32 directed/soft primary assignments and censoring','other_heads':'unchanged59','recipe':'equal average with existing full62 tree, no tuning','external_model':'public BSD3 checkpoint, synthetic pretraining; competition rules section2.6 permits accessible external models'}
def estimator(f):
 return TabICLClassifier(n_estimators=4,batch_size=1,model_path=CHECKPOINT,allow_auto_download=False,device='mps',n_jobs=2,use_amp=False,offload_mode='cpu',inference_config={k:{'safety_factor':.02} for k in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']},random_state=9800+f,verbose=False)
def main():
 assert torch.backends.mps.is_available();torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.65);assert hashlib.file_digest(CHECKPOINT.open('rb'),'sha256').hexdigest()==SHA;(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));(ROOT/'versions.json').write_text(json.dumps({k:importlib.metadata.version(k) for k in ['tabicl','torch','numpy','scikit-learn','scipy','einops','huggingface-hub','psutil']},indent=2));states=state();base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');parts={k:[] for k in ['cat128','tabicl']};audit=[];start=time.time()
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2));pred={k:np.zeros((len(d),2)) for k in parts}
  for f in range(4):
   if fam=='directed_transfer':y,tr,va=labels(d,a,f)
   else:yy,e,_=target(d,f);y=yy[g];tr=e[g];va=fv[g]==f
   assert not(tr&va).any();original=CatBoostClassifier();path=Path('artifacts/evidence_session50_matchup/current')/fam/f'event1_fold{f}.cbm';original.load_model(str(path));importance=original.get_feature_importance(thread_count=2);assert len(importance)==v['x'].shape[1];selected=np.sort(np.lexsort((np.arange(len(importance)),-importance))[:CONFIG['features']]);x=v['x'][:,selected].astype(np.float32);context=ROOT/f'{fam}_fold{f}_context.npz';np.savez_compressed(context,x_train=x[tr],y_train=y[tr].astype(np.int8),x_query=x[va],training_rows=np.flatnonzero(tr),query_rows=np.flatnonzero(va),selected_columns=selected);original_p=original.predict_proba(v['x'][va],thread_count=2)[:,1];np.save(ROOT/f'{fam}_fold{f}_original.npy',original_p)
   m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);dest=ROOT/'cat128';dest.mkdir(exist_ok=True);m.save_model(str(dest/f'{fam}_fold{f}.cbm'));p=m.predict_proba(x[va],thread_count=2)[:,1];pred['cat128'][g[va],actor[va]]=p;np.save(dest/f'{fam}_fold{f}_actions.npy',p);del m;gc.collect();torch.mps.empty_cache();print('TABICL_START',fam,f,'train',int(tr.sum()),'query',int(va.sum()),flush=True)
   m=estimator(f);m.fit(x[tr],y[tr]);p=m.predict_proba(x[va])[:,1];assert np.isfinite(p).all();dest=ROOT/'tabicl';dest.mkdir(exist_ok=True);np.save(dest/f'{fam}_fold{f}_actions.npy',p);pred['tabicl'][g[va],actor[va]]=p;record={'family':fam,'fold':f,'training_actions':int(tr.sum()),'positive_training_actions':int(y[tr].sum()),'query_actions':int(va.sum()),'selected_features':len(selected),'retained_Cat_importance':float(importance[selected].sum()),'selector_model_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'context_sha256':hashlib.file_digest(context.open('rb'),'sha256').hexdigest(),'validation_overlap':0,'elapsed_seconds':time.time()-start};audit.append(record);(ROOT/'fit_audit.json').write_text(json.dumps(audit,indent=2));print('TABICL_DONE',record,flush=True);del m,original,x;gc.collect();torch.mps.empty_cache()
  for k,p in pred.items():parts[k].append(d.select('pair_id','hand_id').with_columns(pl.Series('replacement',(p*dw).sum(1))))
 for k in parts:
  dest=ROOT/k;q=base.join(pl.concat(parts[k]),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('replacement','bg_primary').alias('bg_primary')).drop('replacement');assert len(q)==len(base);q.write_parquet(dest/'event_oof.parquet');assemble(dest)
if __name__=='__main__':main()
