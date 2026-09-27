import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl
from catboost import CatBoostClassifier
root=Path('artifacts/policy');names=['none','directed_transfer','soft_play','coordinated_isolation'];ev=pl.read_csv('data/evaluation_pairs.csv').select('pair_id')
for mode in ['residual','combined']:
 while not (root/f'{mode}_columns.json').exists():time.sleep(3)
 cols=json.loads((root/f'{mode}_columns.json').read_text());models=[]
 for f in range(4):
  m=CatBoostClassifier();m.load_model(str(root/f'{mode}_fold{f}.cbm'));models.append(m)
 parts=[]
 for path in sorted((root/'pair_features').glob('*.parquet')):
  z=pl.read_parquet(path).filter(pl.col('phase')=='evaluation').join(ev,on='pair_id')
  if mode=='combined':
   cx=pl.read_parquet(Path('artifacts/context_pairs')/path.name).filter(pl.col('phase')=='evaluation');add=[c for c in cols if c not in z.columns];z=z.join(cx.select('pair_id',*add),on='pair_id')
  pr=np.mean([m.predict_proba(z.select(cols).to_numpy(),thread_count=3) for m in models],axis=0)
  parts.append(z.select('pair_id').with_columns(*[pl.Series(n,pr[:,k]) for k,n in enumerate(names)]))
 pl.concat(parts).sort('pair_id').write_csv(root/f'{mode}_eval.csv');print('predicted',mode,flush=True)
