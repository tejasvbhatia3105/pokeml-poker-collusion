import json
import numpy as np,polars as pl
from session79_uncertain_targets import ROOT,C
from session12_compare import compare
def main():
 q=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33'),'pressure59','full')
 for kind in ['hard','marginal']:q=q.join(pl.read_parquet(ROOT/kind/'background_oof.parquet').select('pair_id','hand_id',C('cat_and_joint').alias(kind)),on=['pair_id','hand_id'],validate='1:1').with_columns(((C(kind)+C('full'))*.5).alias(kind+'_r33_recipe'))
 q.write_parquet(ROOT/'combined_oof.parquet');names=['r33','pressure59','hard','marginal','hard_r33_recipe','marginal_r33_recipe'];r,_=compare(ROOT/'combined_oof.parquet',names,'uncertain_targets');pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={}
 for n in names:
  delta=pool[n].to_numpy()-pool['r33'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out[n]={'MAP':r[n].mean(),'gain_vs_r33':r[n].mean()-r['r33'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows()),'better':int((r[n]>r['r33']+1e-12).sum()),'worse':int((r[n]<r['r33']-1e-12).sum())}
 (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
