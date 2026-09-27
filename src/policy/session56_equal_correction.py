import json
import numpy as np,polars as pl
from session55_current_targets import ROOT
from session12_compare import compare
C=pl.col
def main():
 folder=ROOT/'correction';q=pl.read_parquet(folder/'comparison_input.parquet').with_columns(((C('r32')+C('independent'))*.5).alias('equal_correction'));q.write_parquet(folder/'equal_input.parquet');r,_=compare(folder/'equal_input.parquet',['r32','independent','equal_correction'],'current_equal');pool=r.group_by('table_id').agg(C('r32').sum(),C('equal_correction').sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));delta=pool['equal_correction'].to_numpy()-pool['r32'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out={'hypothesis':'post-result fixed equal old/refitted correction, no weight search','MAP':r['equal_correction'].mean(),'gain_vs_r32':r['equal_correction'].mean()-r['r32'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C('equal_correction').mean()).sort('fold')['equal_correction'].to_list(),'improved_vs_r32':int((r['equal_correction']>r['r32']+1e-12).sum()),'worse_vs_r32':int((r['equal_correction']<r['r32']-1e-12).sum())};(folder/'equal_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
