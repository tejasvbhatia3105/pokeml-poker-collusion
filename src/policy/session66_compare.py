import json
import numpy as np,polars as pl
from session66_current_candidates import ROOT,C
from session12_compare import compare
def main():
 names=['analytic','compact','grounded','compact_transport','grounded_transport'];q=pl.read_parquet(ROOT/'oof.parquet');q=q.join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33'),'r32'),on=['pair_id','hand_id'],validate='1:1');q.write_parquet(ROOT/'combined_oof.parquet');names=['r32','r33',*names];r,_=compare(ROOT/'combined_oof.parquet',names,'current_candidates');pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={}
 for n in names:
  delta=pool[n].to_numpy()-pool['r33'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out[n]={'MAP':r[n].mean(),'gain_vs_r33':r[n].mean()-r['r33'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows()),'improved_vs_r33':int((r[n]>r['r33']+1e-12).sum()),'worse_vs_r33':int((r[n]<r['r33']-1e-12).sum())}
 (ROOT/'r33_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
