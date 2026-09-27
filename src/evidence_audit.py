import polars as pl
import numpy as np
x=pl.read_parquet('artifacts/dev_hands.parquet').join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id','evidence_rank','behavior_family'),on=['pair_id','hand_id'],how='left')
e=x.filter(pl.col('evidence_rank').is_not_null())
print('evidence rows',len(e))
for b in ['directed_transfer','soft_play','coordinated_isolation']:
    ee=e.filter(pl.col('behavior_family')==b)
    print(b,'time quantiles',ee['time'].quantile(.1),ee['time'].median(),ee['time'].quantile(.9))
    print(ee.group_by('evidence_rank').agg(pl.col('time').mean(),pl.col('pot').mean(),pl.col('both_showdown').mean()).sort('evidence_rank'))
print('first pair',e.filter(pl.col('pair_id')=='P00082F54BA9A').select('hand_id','time','evidence_rank','pot'))
