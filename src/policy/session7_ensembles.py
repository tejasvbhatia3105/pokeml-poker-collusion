\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from session6_priority import inclusion
ROOT=Path('artifacts/evidence_session7');C=pl.col
def main():
    d=pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet').select('pair_id','hand_id','time',C('primary').alias('cat_primary'),C('secondary').alias('cat_secondary'))
    for name,path in [('hist',ROOT/'hist_events_oof.parquet'),('em',ROOT/'em1_oof.parquet')]:
        d=d.join(pl.read_parquet(path).select('pair_id','hand_id',C('primary').alias(name+'_primary'),C('secondary').alias(name+'_secondary'),C('score').alias(name+'_score')),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(pl.read_parquet(ROOT/'hist_eventblend_oof.parquet').select('pair_id','hand_id',C('score').alias('histblend_score')),on=['pair_id','hand_id'],validate='1:1');parts=[]
    for _,g in d.sort('pair_id','time','hand_id').group_by('pair_id',maintain_order=True):
        pri=np.zeros(len(g));sec=np.zeros(len(g))
        for name in ['cat','hist','em']:
            a=g[name+'_primary'].to_numpy();b=g[name+'_secondary'].to_numpy();s=np.maximum(1,a+b);pri+=a/s/3;sec+=b/s/3
        parts.append(g.select('pair_id','hand_id').with_columns(pl.Series('score',inclusion(pri,sec))))
    pl.concat(parts).write_parquet(ROOT/'event_trio_oof.parquet')
    d.select('pair_id','hand_id',((C('em_score')+C('histblend_score'))*.5).alias('score')).write_parquet(ROOT/'event_fusion_oof.parquet')
if __name__=='__main__':main()
