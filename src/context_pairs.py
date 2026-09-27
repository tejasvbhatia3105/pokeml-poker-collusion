import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import polars as pl,numpy as np,time,itertools
from pathlib import Path
from features import build
import glob,io
_native_read=pl.read_parquet
def buffered_read(source,**kwargs):
 if isinstance(source,(str,Path)) and str(source).startswith("artifacts/compact/"):
  paths=sorted(glob.glob(str(source)))
  return pl.concat([_native_read(io.BytesIO(Path(p).read_bytes()),**kwargs) for p in paths])
 return _native_read(source,**kwargs)
pl.read_parquet=buffered_read
from train_baseline import aggregates
labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
ev=pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2')
known=pl.concat([labs,ev]);dest=Path('artifacts/context_pairs');dest.mkdir(exist_ok=True)
t=time.time()
for i,path in enumerate(sorted(Path('artifacts/compact/seats').glob('table_id=*'))):
 table=path.name.split('=')[1];out=dest/f'{table}.parquet'
 if out.exists():continue
 ids=pl.read_parquet(path/'*.parquet',columns=['player_id'])['player_id'].unique().sort().to_list()
 pairs=pl.DataFrame(list(itertools.combinations(ids,2)),schema=['player_1','player_2'],orient='row').join(known,on=['player_1','player_2'],how='left').with_columns(pl.col('pair_id').fill_null(pl.col('player_1')+'_'+pl.col('player_2')))
 h=build(table,pairs,pairs)
 f=aggregates(h).join(pairs,on='pair_id')
                                                             
 cols=[c for c in f.columns if c.endswith(('_mean','_rate','_top5'))]
 for role in ['player_1','player_2']:
  endpoints=pl.concat([f.select('phase',pl.col(r).alias('player'),*cols,'n_hands') for r in ['player_1','player_2']])
  marg=endpoints.group_by('phase','player').agg(*[(pl.col(c)*pl.col('n_hands')).sum().truediv(pl.col('n_hands').sum()).alias(c+'_'+role) for c in cols])
  f=f.join(marg,left_on=['phase',role],right_on=['phase','player'])
 expr=[]
 for c in cols:
  x=pl.col(c);a=pl.col(c+'_player_1');b=pl.col(c+'_player_2');mu=x.mean().over('phase')
  expected=a+b-mu
  expr += [(x-expected).cast(pl.Float32).alias(c+'_excess'),((x-expected)/(expected.abs()+.1)).cast(pl.Float32).alias(c+'_relative'),pl.max_horizontal(a,b).cast(pl.Float32).alias(c+'_player_max')]
 f=f.with_columns(expr).drop([c+'_'+role for c in cols for role in ['player_1','player_2']])
 f.write_parquet(out,compression='zstd')
 if i%20==0:print(i,round(time.time()-t,1),flush=True)
