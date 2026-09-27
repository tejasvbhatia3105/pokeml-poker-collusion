\
\
\
\
import json
from pathlib import Path
import numpy as np,polars as pl
from session12_compare import compare
ROOT=Path('artifacts/evidence_session64_equal_list_pressure');C=pl.col
def main():
 ROOT.mkdir(exist_ok=True);q=pl.read_parquet('artifacts/evidence_session62_grounded_list_boost/combined_oof.parquet').select('pair_id','hand_id','r32','pressure59','full');q=q.with_columns(((C('pressure59')+C('full'))*.5).alias('equal'));q.write_parquet(ROOT/'oof.parquet');names=['r32','pressure59','full','equal'];r,_=compare(ROOT/'oof.parquet',names,'equal_list_pressure');pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={'method':__doc__,'MAP':r['equal'].mean(),'folds':r.group_by('fold').agg(C('equal').mean()).sort('fold')['equal'].to_list(),'comparisons':{}}
 for n in ['r32','pressure59']:
  delta=pool['equal'].to_numpy()-pool[n].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out['comparisons'][n]={'gain':r['equal'].mean()-r[n].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'improved':int((r['equal']>r[n]+1e-12).sum()),'worse':int((r['equal']<r[n]-1e-12).sum())}
 (ROOT/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
