\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session51_matchup_inference as baseline
from session5_multiway_features import build
from session59_pressure_equity import states,canonical,equity,fields
from session27_donor_call_witness import noisy_or
ROOT=Path('artifacts/evidence_session63_pressure_inference');MODEL=Path('artifacts/evidence_session59_pressure_equity');C=pl.col
def load_models():
 out=baseline.load_models();out['pressure_cfg']=json.load(open(MODEL/'config.json'));out['pressure']=[]
 for f in range(4):
  models=[]
  for k in [1,2]:m=CatBoostClassifier();m.load_model(str(MODEL/f'event{k}_fold{f}_em2.cbm'));models.append(m)
  out['pressure'].append(models)
 return out
def design(d,players,cfg):
 assert d['table_id'].n_unique()==1;table=d['table_id'][0];query=d.select('pair_id','hand_id').join(players.select('pair_id','player_1','player_2'),on='pair_id',validate='m:1');raw=build(table,query,return_actions=True);raw=raw.filter((C('action_class')==3)&~C('facing_partner')&C('mw_alive')&(C('players_active')>=3));ac=cfg['action_columns'];a=raw.select('pair_id','hand_id',*ac).join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='m:1').sort('row','street_no','action_no');assert a.select('pair_id','hand_id','street_no','action_no').n_unique()==len(a);hc=cfg['hand_columns'];g=a['row'].to_numpy()
 if len(a):
  ss,mm,meta=states(d,a,players);cs=np.array([canonical(s) for s in ss],np.int8);packed=np.column_stack([cs,mm]);unique,inverse=np.unique(packed,axis=0,return_inverse=True);ex=fields(equity(unique[:,:17],unique[:,17],4096)[inverse],meta,a)
 else:ex=np.empty((0,7),np.float32)
 x=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],ex]);return a,x,ex
def pressure_events(d,players,models,folds=range(4)):
 d=d.sort('pair_id','time','hand_id').drop('row',strict=False).with_row_index('row');a,x,_=design(d,players,models['pressure_cfg']);g=a['row'].to_numpy();q=d.select('pair_id','hand_id')
 for f in folds:
  for k in range(2):
   p=models['pressure'][f][k].predict_proba(x,thread_count=2)[:,1] if len(a) else np.empty(0);q=q.with_columns(pl.Series(f'pressure_{k+1}_{f}',noisy_or(p,g,len(d))))
 return q
def events(d,players,models,folds=range(4),preserve_baseline=False):
 folds=list(folds);z=baseline.events(d,players,models,folds)
 if preserve_baseline:z=z.with_columns(*[C(f'new_{head}_{f}').alias(f'list_{head}_{f}') for f in folds for head in ['primary','secondary']])
 iso=z.filter(C('behavior_family')=='coordinated_isolation')
 if len(iso):
  q=pressure_events(iso,players,models,folds);z=z.join(q,on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left');drop=[]
  for f in folds:
   for k,head in [(1,'primary'),(2,'secondary')]:
    source=f'pressure_{k}_{f}';z=z.with_columns(pl.coalesce(source,f'new_{head}_{f}').alias(f'new_{head}_{f}'));drop.append(source)
  z=z.drop(drop)
 return z
