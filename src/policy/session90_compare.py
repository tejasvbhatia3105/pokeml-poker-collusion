import json
import numpy as np,polars as pl
import session89_compare as compare
from session90_partial_crop_list import ROOT
C=pl.col
def main():
 for arm in ['complete','all']:
  compare.ROOT=ROOT/arm;compare.main()
 a=pl.read_parquet(ROOT/'complete/pair_comparison.parquet').select('pair_id','table_id','window',C('candidate').alias('complete'));b=pl.read_parquet(ROOT/'all/pair_comparison.parquet').select('pair_id','window',C('candidate').alias('all'));q=a.join(b,on=['pair_id','window'],validate='1:1');out={}
 for (w,),g in q.group_by('window'):
  pool=g.group_by('table_id').agg(C('complete','all').sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(9012).integers(0,len(pool),(5000,len(pool)));delta=pool['all'].to_numpy()-pool['complete'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out[w]={'all_minus_complete':g['all'].mean()-g['complete'].mean(),'CI95':np.quantile(boot,[.025,.975]).tolist(),'pairs':len(g)}
 (ROOT/'partial_vs_complete.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
