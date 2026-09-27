import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,pandas as pd,polars as pl
from sklearn.metrics import average_precision_score
root=Path('artifacts/policy');old=pd.read_csv('artifacts/selected_pair_oof.csv').sort_values('pair_id');new=pd.read_csv(root/'residual_oof.csv').sort_values('pair_id');assert old.pair_id.tolist()==new.pair_id.tolist()
y=new.truth.to_numpy()>0;a=1-old.none.to_numpy();b=1-new.none.to_numpy();meta=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='development').select('pair_id','table_id').collect().to_pandas();g=new[['pair_id']].merge(meta,on='pair_id').table_id.to_numpy();tables=np.unique(g);rng=np.random.default_rng(1401);draws=[]
for i in range(1000):
 ix=rng.integers(0,len(tables),len(tables));counts=np.bincount(ix,minlength=len(tables));weights=counts[np.searchsorted(tables,g)]*np.where(y,1,50)
 draws.append(average_precision_score(y,b,sample_weight=weights)-average_precision_score(y,a,sample_weight=weights))
report={'rare_target_negative_weight':50,'old_ap':average_precision_score(y,a,sample_weight=np.where(y,1,50)),'new_ap':average_precision_score(y,b,sample_weight=np.where(y,1,50)),'paired_pool_bootstrap_delta_95pct':np.quantile(draws,[.025,.975]).tolist(),'bootstrap_replicates':1000,'limitation':'Fixed OOF predictions, resampling pools; does not quantify private-test distribution shift or model-selection uncertainty.'}
(root/'comparison.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
