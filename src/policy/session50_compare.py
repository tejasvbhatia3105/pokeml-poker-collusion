import json
import numpy as np,polars as pl
from session50_matchup import ROOT
from session12_compare import compare
def main():
 paths={'r31':'artifacts/evidence_session41_isolation_bet_fold/paired_fold','full':str(ROOT),'outcomes':str(ROOT/'outcomes'),'current':str(ROOT/'current')};d=None
 for name,path in paths.items():
  q=pl.read_parquet(path+'/background_oof.parquet').select('pair_id','hand_id',pl.col('cat_and_joint').alias(name));d=q if d is None else d.join(q,on=['pair_id','hand_id'],validate='1:1')
 d.write_parquet(ROOT/'combined_oof.parquet');r,report=compare(ROOT/'combined_oof.parquet',list(paths),'matchup_combined');pool=r.group_by('table_id').agg(pl.col(list(paths)).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={}
 for name in paths:
  delta=pool[name].to_numpy()-pool['r31'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out[name]={'MAP':r[name].mean(),'gain_vs_r31':r[name].mean()-r['r31'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'improved_vs_r31':int((r[name]>r['r31']+1e-12).sum()),'worse_vs_r31':int((r[name]<r['r31']-1e-12).sum())}
 (ROOT/'r31_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
