import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import polars as pl
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session25_persistent_actor');C=pl.col
def main():
 base=pl.read_parquet(ROOT/'oriented/event_oof.parquet');cond=pl.read_parquet(ROOT/'conditional_oof.parquet');prob=pl.read_parquet(ROOT/'actor_oof.parquet').select('pair_id','actor0','actor1');outs=[]
 for r in range(2):
  root=ROOT/f'actor{r}';root.mkdir(exist_ok=True);q=base.join(cond.select('pair_id','hand_id',C(f'actor{r}_event0').alias('new1'),C(f'actor{r}_event1').alias('new2')),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new1','bg_primary').alias('bg_primary'),pl.coalesce('new2','bg_secondary').alias('bg_secondary')).drop('new1','new2');q.write_parquet(root/'event_oof.parquet');assemble(root);outs.append(pl.read_parquet(root/'background_oof.parquet'))
 q=outs[0].join(outs[1],on=['pair_id','hand_id'],suffix='_1',validate='1:1').join(prob,on='pair_id',how='left',validate='m:1').with_columns(C('actor0').fill_null(.5),C('actor1').fill_null(.5))
 q.select('pair_id','hand_id',*[(C(n)*C('actor0')+C(n+'_1')*C('actor1')).alias(o) for n,o in [('cat_only','cat_persistent'),('cat_and_joint','joint_persistent'),('propagated','propagated_persistent')]]).write_parquet(ROOT/'persistent_oof.parquet')
if __name__=='__main__':main()
