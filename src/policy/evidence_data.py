import json,polars as pl
from pathlib import Path
from sequence_features import augment
C=pl.col
ROOT=Path('artifacts/policy')
def load():
 cache=ROOT/'evidence_training.parquet';cols=json.loads((ROOT/'sequence/evidence_columns.json').read_text())
 if cache.exists():return pl.read_parquet(cache),cols
 labs=pl.read_csv('data/development_labels.csv').filter(C('label')==1).select('pair_id','behavior_family')
 h=pl.scan_parquet(str(ROOT/'hand_features/*.parquet')).filter(C('phase')=='development').join(labs.lazy().select('pair_id'),on='pair_id').collect().sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')]
 h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')]
 h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
 d=pl.read_parquet('artifacts/dev_detail.parquet').join(labs,on='pair_id').join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']).join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0),(C('time')/.6).alias('relative_time'))
 d,_=augment(d);d=d.sort('pair_id','hand_id');d.write_parquet(cache,compression='zstd');return d,cols
