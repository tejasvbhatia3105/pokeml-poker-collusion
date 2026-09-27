import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');sys.path.insert(0,'src')
from pathlib import Path
import json,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
from train_baseline import ap
from sequence_features import augment
root=Path('artifacts/policy');C=pl.col;names=['none','directed_transfer','soft_play','coordinated_isolation'];pair=pd.read_csv(root/'residual_oof.csv').sort_values('pair_id');pair['family']=pair[names[1:]].idxmax(axis=1);pair['risk_score']=1-pair.none
h=pl.scan_parquet(str(root/'hand_features/*.parquet')).filter(C('phase')=='development').join(pl.from_pandas(pair.loc[pair.truth>0,['pair_id']]).lazy(),on='pair_id').collect().sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')]
h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')];h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
d=pl.read_parquet('artifacts/dev_detail.parquet').join(pl.from_pandas(pair.loc[pair.truth>0,['pair_id','family']]),on='pair_id').join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']).with_columns((C('time')/.6).alias('relative_time'));d,_=augment(d)
oc=json.loads(Path('artifacts/rank_columns.json').read_text());nc=json.loads((root/'sequence/evidence_columns.json').read_text());folds=json.loads(Path('artifacts/folds.json').read_text());parts=[]
for f in folds:
 for n in names[1:]:
  z=d.filter(C('table_id').is_in(f['valid_tables'])&(C('family')==n))
  if not len(z):continue
  m=CatBoostClassifier();m.load_model(f'artifacts/rank_{n}_fold{f["fold"]}.cbm');op=m.predict_proba(z.select(oc).to_numpy(),thread_count=3)[:,1]
  m.load_model(str(root/f'sequence/evidence_{n}_fold{f["fold"]}.cbm'));npred=m.predict_proba(z.select(nc).to_numpy(),thread_count=3)[:,1]
  parts.append(z.select('pair_id','hand_id').with_columns(pl.Series('score',.25*op+.75*npred)))
preds=pl.concat(parts).sort(['pair_id','score','hand_id'],descending=[False,True,False]);truth={}
preds.group_by('pair_id',maintain_order=True).agg(C('hand_id').head(5)).write_parquet(root/'combined_selected_evidence.parquet')
for pid,hand in pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').iter_rows():truth.setdefault(pid,set()).add(hand)
records=[]
for key,g in preds.group_by('pair_id',maintain_order=True):
 pid=key[0];rel=np.array([h in truth[pid] for h in g['hand_id'][:5]]);score=float((np.cumsum(rel)/np.arange(1,len(rel)+1)*rel).sum()/min(5,len(truth[pid])));records.append({'pair_id':pid,'map5':score})
risk=pair.risk_score.to_numpy();y=pair.truth.to_numpy();family=pair.family.to_numpy();pap=ap(y>0,risk);bm=np.mean([ap(y==k,risk*(family==n)*(risk>=.01)) for k,n in enumerate(names[1:],1)]);emap=np.mean([r['map5'] for r in records]);report={'pair_ap':pap,'evidence_map5':emap,'behavior_map':bm,'combined_score':.7*pap+.2*emap+.1*bm,'validation':'Disjoint player-pool development OOF, selected iteration and evidence blend on these folds; no fourth-family labels or private test targets.'}
(root/'combined_validation.json').write_text(json.dumps(report,indent=2));pd.DataFrame(records).to_csv(root/'combined_evidence_validation.csv',index=False);print(report,flush=True)
