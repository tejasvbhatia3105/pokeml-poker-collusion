import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import polars as pl
from session12_event_replacement import assemble
C=pl.col
def main():
 root=Path('artifacts/evidence_session20_fixed_ensemble');root.mkdir(exist_ok=True);q=pl.concat([pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).select('pair_id','hand_id',C('cat_primary').alias('p0'),C('cat_secondary').alias('s0')) for f in range(4)]);paths=['artifacts/evidence_session13_actor/global/event_oof.parquet','artifacts/evidence_session14_outcomes/factual/event_oof.parquet','artifacts/evidence_session12/unlabelled_gated/event_oof.parquet']
 for i,path in enumerate(paths,1):q=q.join(pl.read_parquet(path).select('pair_id','hand_id',C('bg_primary').alias(f'p{i}'),C('bg_secondary').alias(f's{i}')),on=['pair_id','hand_id'],validate='1:1')
 q.select('pair_id','hand_id',pl.mean_horizontal([f'p{i}' for i in range(4)]).alias('bg_primary'),pl.mean_horizontal([f's{i}' for i in range(4)]).alias('bg_secondary')).write_parquet(root/'event_oof.parquet');(root/'config.json').write_text(json.dumps({'weights':[.25]*4,'members':['original Cat']+paths,'selection':'post-result fixed equal blend of three independently useful mechanisms plus original; no tuned family splice or weights'},indent=2));assemble(root)
 root=Path('artifacts/evidence_session21_em_revisit');root.mkdir(exist_ok=True);pl.read_parquet('artifacts/evidence_session7/em1_oof.parquet').select('pair_id','hand_id',C('primary').alias('bg_primary'),C('secondary').alias('bg_secondary')).write_parquet(root/'event_oof.parquet');(root/'config.json').write_text(json.dumps({'source':'artifacts/evidence_session7/em1_oof.parquet','hypothesis':'The old EM1 model improved standalone Cat inclusion but was rejected with R28; test its interaction with the later R30 conditional correction','no_retraining':True},indent=2));assemble(root)
if __name__=='__main__':main()
