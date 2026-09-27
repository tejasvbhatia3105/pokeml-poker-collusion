import json
import numpy as np,polars as pl
from session55_current_targets import ROOT
from session12_compare import compare
def main():
 folder=ROOT/'correction';q=pl.read_parquet(folder/'list_learning_oof.parquet').select('pair_id','hand_id','independent','independent_inclusion');base=pl.read_parquet('artifacts/evidence_session50_matchup/current/background_oof.parquet').select('pair_id','hand_id',pl.col('cat_and_joint').alias('r32'));q=q.join(base,on=['pair_id','hand_id'],validate='1:1');q.write_parquet(folder/'comparison_input.parquet');names=['r32','independent','independent_inclusion'];r,_=compare(folder/'comparison_input.parquet',names,'current_refit');pool=r.group_by('table_id').agg(pl.col(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={}
 for name in names:
  delta=pool[name].to_numpy()-pool['r32'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out[name]={'MAP':r[name].mean(),'gain_vs_r32':r[name].mean()-r['r32'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'improved_vs_r32':int((r[name]>r['r32']+1e-12).sum()),'worse_vs_r32':int((r[name]<r['r32']-1e-12).sum())}
 (folder/'r32_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
