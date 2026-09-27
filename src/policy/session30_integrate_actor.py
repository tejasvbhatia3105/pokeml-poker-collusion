\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import polars as pl
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session30_actor_integration')
def main():
 ROOT.mkdir(exist_ok=True);paths=['artifacts/evidence_session26_exact_fold/action_hand/event_oof.parquet','artifacts/evidence_session13_actor/global/event_oof.parquet','artifacts/evidence_session14_outcomes/factual/event_oof.parquet','artifacts/evidence_session12/unlabelled_gated/event_oof.parquet'];q=None
 for i,path in enumerate(paths):
  z=pl.read_parquet(path).select('pair_id','hand_id',C('bg_primary').alias(f'p{i}'),C('bg_secondary').alias(f's{i}'));q=z if q is None else q.join(z,on=['pair_id','hand_id'],validate='1:1')
 q.select('pair_id','hand_id',pl.mean_horizontal([f'p{i}' for i in range(4)]).alias('bg_primary'),pl.mean_horizontal([f's{i}' for i in range(4)]).alias('bg_secondary')).write_parquet(ROOT/'event_oof.parquet');(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'weights':[.25]*4,'members':paths},indent=2));assemble(ROOT)
if __name__=='__main__':main()
