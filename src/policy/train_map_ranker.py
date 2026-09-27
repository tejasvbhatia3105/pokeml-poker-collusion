\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostRanker,Pool
from evidence_data import load
root=Path('artifacts/policy');dest=root/'map_ranker';dest.mkdir(exist_ok=True);d,cols=load();folds=json.loads(Path('artifacts/folds.json').read_text());parts=[];t=time.time()
for f in folds:
 for behavior in ['directed_transfer','soft_play','coordinated_isolation']:
  z=d.filter(pl.col('behavior_family')==behavior);X=z.select(cols).to_numpy();y=z['evidence'].to_numpy();g=z['pair_id'].to_numpy();va=z['table_id'].is_in(f['valid_tables']).to_numpy();tr=~va
  train=Pool(X[tr],label=y[tr],group_id=g[tr]);valid=Pool(X[va],label=y[va],group_id=g[va]);m=CatBoostRanker(iterations=800,depth=5,learning_rate=.035,loss_function='YetiRank:mode=MAP;top=5',eval_metric='MAP:top=5',l2_leaf_reg=10,thread_count=5,random_seed=1901+f['fold'],verbose=False,allow_writing_files=False);m.fit(train,eval_set=valid,early_stopping_rounds=150);m.save_model(str(dest/f'{behavior}_fold{f["fold"]}.cbm'));parts.append(z.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence').with_columns(pl.Series('ranker_score',m.predict(X[va]))));print('fit',f['fold'],behavior,m.tree_count_,'seconds',round(time.time()-t,1),flush=True)
scores=pl.concat(parts).join(pl.read_parquet(root/'sequence/evidence_oof.parquet').select('pair_id','hand_id',(.25*pl.col('old_score')+.75*pl.col('new_score')).alias('v4_score')),on=['pair_id','hand_id']);scores=scores.sort('pair_id','hand_id').with_columns(pl.col('v4_score').rank('ordinal',descending=True).over('pair_id').alias('v4_rank'),pl.col('ranker_score').rank('ordinal',descending=True).over('pair_id').alias('new_rank'));scores.write_parquet(dest/'oof.parquet');rows=[]
for alpha in [0,.25,.5,.75,1]:
 z=scores.with_columns(((1-alpha)/(5+pl.col('v4_rank'))+alpha/(5+pl.col('new_rank'))).alias('score')).sort(['pair_id','score','hand_id'],descending=[False,True,False])
 for key,g in z.group_by('pair_id',maintain_order=True):
  rel=g['evidence'].to_numpy()[:5];value=(np.cumsum(rel)/np.arange(1,len(rel)+1)*rel).sum()/min(5,g['evidence'].sum());rows.append({'ranker_weight':alpha,'pair_id':key[0],'behavior':g['behavior_family'][0],'map5':value})
f=pd.DataFrame(rows);f.to_csv(dest/'validation.csv',index=False);print('MAP',f.groupby('ranker_weight').map5.mean().to_dict(),flush=True);print(f.groupby(['ranker_weight','behavior']).map5.mean().to_string(),flush=True)
