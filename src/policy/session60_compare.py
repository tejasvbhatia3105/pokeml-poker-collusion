import json
import numpy as np,polars as pl
from session60_aligned_multiway import ROOT,C
from session12_compare import compare
def main():
 d=pl.read_parquet('artifacts/evidence_session50_matchup/current/background_oof.parquet').select('pair_id','hand_id',C('cat_and_joint').alias('r32'));d=d.join(pl.read_parquet('artifacts/evidence_session59_pressure_equity/background_oof.parquet').select('pair_id','hand_id',C('cat_and_joint').alias('pressure59')),on=['pair_id','hand_id'],validate='1:1');cols=['cat_only','cat_and_joint','propagated'];d=d.join(pl.read_parquet(ROOT/'background_oof.parquet').select('pair_id','hand_id',*cols),on=['pair_id','hand_id'],validate='1:1');d.write_parquet(ROOT/'combined_oof.parquet');names=['r32','pressure59',*cols];r,_=compare(ROOT/'combined_oof.parquet',names,'aligned_precise');pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={}
 for n in names:
  delta=pool[n].to_numpy()-pool['r32'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out[n]={'MAP':r[n].mean(),'gain_vs_r32':r[n].mean()-r['r32'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows()),'improved_vs_r32':int((r[n]>r['r32']+1e-12).sum()),'worse_vs_r32':int((r[n]<r['r32']-1e-12).sum())}
 (ROOT/'r32_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
