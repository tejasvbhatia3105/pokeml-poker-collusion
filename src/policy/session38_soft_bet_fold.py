import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session29_shared_fold_witness import target
from session37_bet_fold import paired_features
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session38_soft_bet_fold');BASE=Path('artifacts/evidence_session37_bet_fold/paired_hand')
def data():
 d=hand_data().filter(C('behavior_family')=='soft_play').drop('row').with_row_index('row');old=Path('artifacts/evidence_session29_shared_folds');a=pl.read_parquet(old/'fold_actions.parquet').drop('row').join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='m:1').with_row_index('action_row');assert a.select('pair_id','hand_id').n_unique()==len(a);cfg=json.load(open(old/'columns.json'));return d,a,cfg
def main():
 ROOT.mkdir(exist_ok=True);d,a,cfg=data();a.write_parquet(ROOT/'fold_actions.parquet');extra,alignment=paired_features(d,a);extra.write_parquet(ROOT/'action_features.parquet');alignment.write_parquet(ROOT/'alignment.parquet');cols=[c for c in extra.columns if c!='action_row'];ex=extra.select(cols).to_numpy();g=a['row'].to_numpy();fv=d['fold'].to_numpy();x=np.column_stack([a.select(cfg['action']).to_numpy(),d.select(cfg['hand']).to_numpy()[g]]);pred={k:np.zeros(len(d)) for k in ['original_soft','paired_soft']};audit=[];start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'fold_columns':cfg['action'],'hand_columns':cfg['hand'],'paired_columns':cols,'baseline':str(BASE),'target':'outer-training session17 grounded soft fold tier; both endpoint roles permitted; unique fold per hand','arms':'original raw+hand control vs same plus exact preceding partner bet context','models':'Cat400 depth5 original primary seed; no pooling with directed family'},indent=2))
 for f in range(4):
  y,e,_=target(d,f);tr=e[g];va=fv[g]==f;assert y.sum()==y[g[tr]].sum();assert not (tr&va).any()
  for kind in pred:
   root=ROOT/kind;root.mkdir(exist_ok=True);xx=x if kind=='original_soft' else np.column_stack([x,ex]);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(xx[tr],y[g[tr]]);m.save_model(str(root/f'primary_fold{f}.cbm'));pred[kind][g[va]]=m.predict_proba(xx[va],thread_count=2)[:,1];audit.append({'fold':f,'kind':kind,'training_actions':int(tr.sum()),'positive_actions':int(y[g[tr]].sum()),'heldout_actions':int(va.sum()),'overlap':0});print('soft bet fold',f,kind,round(time.time()-start,1),flush=True)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));base=pl.read_parquet(BASE/'event_oof.parquet')
 for kind,pp in pred.items():
  root=ROOT/kind;q=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',pp));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary')).drop('new_primary').write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
