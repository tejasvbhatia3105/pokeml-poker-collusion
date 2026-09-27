import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import polars as pl,json,time
from pathlib import Path
r=Path('artifacts/policy')
while len(list((r/'pair_features').glob('*.parquet')))<400:time.sleep(3)
a=pl.scan_parquet(str(r/'actions/*.parquet'))
f=pl.scan_parquet(str(r/'pair_features/*.parquet'))
report={'action_rows':a.select(pl.len()).collect().item(),'tables':len(list((r/'pair_features').glob('*.parquet')))}
for phase,source in [('development','development_labels'),('evaluation','evaluation_pairs')]:
 keys=pl.read_csv(f'data/{source}.csv').select('pair_id')
 present=f.filter(pl.col('phase')==phase).select('pair_id').collect()
 report[phase+'_missing_pairs']=keys.join(present,on='pair_id',how='anti').height
report['nonfinite_features']=f.select(pl.selectors.float().is_finite().not_().sum()).collect().sum_horizontal().sum()
print(report);(r/'coverage.json').write_text(json.dumps(report,indent=2))
