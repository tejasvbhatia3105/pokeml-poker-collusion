\
\
\
\
import json
from pathlib import Path
import polars as pl
ROOT=Path('artifacts/evidence_session32_fixed_hypotheses');C=pl.col
def main():
 ROOT.mkdir(exist_ok=True);paths=['artifacts/evidence_session30_actor_integration/background_oof.parquet','artifacts/evidence_session31_conditional_events/background_oof.parquet'];a=pl.read_parquet(paths[0]).select('pair_id','hand_id',C('cat_and_joint').alias('actor'));b=pl.read_parquet(paths[1]).select('pair_id','hand_id',C('cat_and_joint').alias('cascade'));a.join(b,on=['pair_id','hand_id'],validate='1:1').with_columns(((C('actor')+C('cascade'))*.5).alias('equal_hypotheses')).write_parquet(ROOT/'oof.parquet');(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'members':paths,'columns':['cat_and_joint']*2,'weights':[.5,.5]},indent=2))
if __name__=='__main__':main()
