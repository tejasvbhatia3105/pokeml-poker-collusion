import json
from pathlib import Path
import numpy as np,polars as pl
from session12_compare import compare
C=pl.col
def main():
 for folder,name in [('evidence_session80_soft_passivity','passivity'),('evidence_session81_conditional_secondary','conditional_secondary')]:
  root=Path('artifacts')/folder;q=pl.read_parquet(root/'background_oof.parquet').select('pair_id','hand_id',C('cat_and_joint').alias(name));q=q.join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33'),'pressure59','full'),on=['pair_id','hand_id'],validate='1:1').with_columns(((C(name)+C('full'))*.5).alias(name+'_r33_recipe'));q.write_parquet(root/'combined_oof.parquet');names=['r33','pressure59',name,name+'_r33_recipe'];r,_=compare(root/'combined_oof.parquet',names,name);pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={}
  for n in names:
   delta=pool[n].to_numpy()-pool['r33'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out[n]={'MAP':r[n].mean(),'gain_vs_r33':r[n].mean()-r['r33'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows()),'better':int((r[n]>r['r33']+1e-12).sum()),'worse':int((r[n]<r['r33']-1e-12).sum())}
  (root/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
