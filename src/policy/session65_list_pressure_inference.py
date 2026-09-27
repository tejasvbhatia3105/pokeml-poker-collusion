import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
import session63_pressure_inference as pressure
from session46_paired_inference import action_design,NAMES,score_pair as pressure_score
from session51_matchup_inference import current_from_design
from session59_pressure_equity import COLS as PRECISION_COLS
from session11_conditional_family import features
from session6_priority import inclusion
from session8_count_conditioning import conditioned
ROOT=Path('artifacts/evidence_session65_list_pressure_inference');TREE=Path('artifacts/evidence_session62_grounded_list_boost');C=pl.col
GROUND_COLUMNS=json.load(open(TREE/'grounded_columns.json'))
EXTRA_COLUMNS=GROUND_COLUMNS+[f'list_{head}_{f}' for f in range(4) for head in ['primary','secondary']]
def load_models():
 out=pressure.load_models();out['list_boost']=[joblib.load(TREE/f'list_boost_full_fold{f}.joblib') for f in range(4)];return out
def grounded(d,players,models):
 a,x,_,_=action_design(d,players,models['cfg']);ac=models['cfg']['fold_columns'];ec=models['cfg']['paired_columns'];raw=np.full((len(d),len(ac)+len(ec)+1),-2,np.float32);raw[:,-1]=0;g=a['row'].to_numpy();raw[g,:len(ac)]=a.select(ac).to_numpy();raw[g,len(ac):-1]=x[:,len(ac)+len(models['cfg']['hand_columns']):];raw[g,-1]=1;current=np.full((len(d),2),-2,np.float32);current[g]=current_from_design(a,x,models['cfg']);p=np.full((len(d),15),-2,np.float32);p[:,-1]=0;iso=d.filter(C('behavior_family')=='coordinated_isolation')
 if len(iso):
  iso=iso.drop('row').with_row_index('row');pa,_,ex=pressure.design(iso,players,models['pressure_cfg']);q=pa.select('pair_id','hand_id').with_columns(*[pl.Series(c,ex[:,i]) for i,c in enumerate(PRECISION_COLS)]);cols=[c+'_'+s for s in ['mean','max'] for c in PRECISION_COLS]+['pressure_count'];q=q.group_by('pair_id','hand_id').agg(*[getattr(C(c),s)().alias(c+'_'+s) for s in ['mean','max'] for c in PRECISION_COLS],pl.len().alias('pressure_count'));q=d.select('pair_id','hand_id').join(q,on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(C('pressure_count').fill_null(0));p=q.select(cols).fill_null(-2).to_numpy().astype(np.float32)
 return np.column_stack([raw,current,p]).astype(np.float32)
def events(d,players,models,folds=range(4)):
 z=pressure.events(d,players,models,folds,preserve_baseline=True);x=grounded(z,players,models);assert x.shape[1]==len(GROUND_COLUMNS);return z.with_columns(*[pl.Series(c,x[:,i]) for i,c in enumerate(GROUND_COLUMNS)])
def tree_delta(model,x):
 a=np.zeros((len(x),2),np.float32)
 for tree,step in model['steps']:a=np.clip(a+step*tree.predict(x),-model['config']['bound'],model['config']['bound']).astype(np.float32)
 return a
def tree_score(g,f,models):
 q=g.with_columns(*[C(f'{n}_{f}').alias(n) for n in NAMES]).with_columns(C(f'list_primary_{f}').alias('cat_primary'),C(f'list_secondary_{f}').alias('cat_secondary'));ca=q.select('cat_primary','cat_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=q.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];ci=inclusion(*ca.T);ji=inclusion(*((ca+hp)/2).T);q=q.with_columns(pl.Series('cat_inclusion',ci),pl.Series('joint_inclusion',ji),pl.Series('r29',.25*q['base'].to_numpy()+.25*ci+.5*ji));x,prior=features(q);x=np.nan_to_num(np.column_stack([x,q.select(GROUND_COLUMNS).to_numpy()]),nan=0,posinf=1e6,neginf=-1e6);model=models['list_boost'][f];z=prior+tree_delta(model,x);p=torch.softmax(torch.tensor(np.column_stack([np.zeros(len(g),np.float32),z])),1).numpy();inc=conditioned(p[:,1:],model['minimums'][g['behavior_family'][0]]);return .25*q['base'].to_numpy()+.25*ci+.5*inc
def score_pair(g,f,models):return .5*pressure_score(g,f,models)+.5*tree_score(g,f,models)
