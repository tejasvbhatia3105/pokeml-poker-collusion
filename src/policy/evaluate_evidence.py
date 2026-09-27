import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import time,json,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
root=Path('artifacts/policy')
while len(list((root/'hand_features').glob('*.parquet')))<400:time.sleep(3)
labs=pl.read_csv('data/development_labels.csv').filter(pl.col('label')==1).select('pair_id','behavior_family')
p=pl.scan_parquet(str(root/'hand_features/*.parquet')).filter(pl.col('phase')=='development').join(labs.lazy().select('pair_id'),on='pair_id').collect().sort('pair_id','time_index')
rcols=[c for c in p.columns if c.endswith('_r')]
p=p.with_columns(*[(pl.col(c)/(pl.col(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rcols])
hz=[c for c in p.columns if c.endswith('_hz')]
p=p.with_columns(*[pl.col(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz])
newcols=[c for c in p.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
d=pl.read_parquet('artifacts/dev_detail.parquet').join(labs,on='pair_id').join(p.select('pair_id','hand_id',*newcols),on=['pair_id','hand_id']).join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(pl.col('evidence').fill_null(0),(pl.col('time')/.6).alias('relative_time')).sort('pair_id','hand_id')
cols=json.loads(Path('artifacts/rank_columns.json').read_text())+newcols
(root/'evidence_columns.json').write_text(json.dumps(cols));folds=json.loads(Path('artifacts/folds.json').read_text());parts=[]
for f in folds:
 for behavior in ['directed_transfer','soft_play','coordinated_isolation']:
  z=d.filter(pl.col('behavior_family')==behavior);X=z.select(cols).to_numpy();y=z['evidence'].to_numpy();va=z['table_id'].is_in(f['valid_tables']).to_numpy();tr=~va
  m=CatBoostClassifier(iterations=850,depth=5,learning_rate=.035,loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=8,thread_count=5,random_seed=710+f['fold'],verbose=False,allow_writing_files=False)
  m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=120);m.save_model(str(root/f'evidence_{behavior}_fold{f["fold"]}.cbm'))
  parts.append(z.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence').with_columns(pl.Series('new_score',m.predict_proba(X[va])[:,1])))
  print('fit',f['fold'],behavior,m.tree_count_,flush=True)
result=pl.concat(parts).join(pl.read_parquet('artifacts/rank_oof.parquet').select('pair_id','hand_id',pl.col('score').alias('old_score')),on=['pair_id','hand_id']);result.write_parquet(root/'evidence_oof.parquet')
rows=[]
for w in [0,.25,.5,.75,1]:
 scores=result.with_columns(((1-w)*pl.col('old_score')+w*pl.col('new_score')).alias('score')).sort(['pair_id','score','hand_id'],descending=[False,True,False])
 for key,g in scores.group_by('pair_id',maintain_order=True):
  relevance=g['evidence'].to_numpy()[:5];ap=float((np.cumsum(relevance)/np.arange(1,len(relevance)+1)*relevance).sum()/min(5,g['evidence'].sum()))
  rows.append({'weight':w,'pair_id':key[0],'behavior':g['behavior_family'][0],'map5':ap})
pd.DataFrame(rows).to_csv(root/'evidence_validation.csv',index=False)
print(pd.DataFrame(rows).groupby(['weight','behavior']).map5.mean().to_string(),flush=True)
print('OVERALL',pd.DataFrame(rows).groupby('weight').map5.mean().to_dict(),flush=True)
