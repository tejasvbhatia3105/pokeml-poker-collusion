import os,json
from pathlib import Path
import numpy as np, polars as pl
from session12_compare import compare
C=pl.col
def main():
 base=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33'),'pressure59','full')
 for session in map(int,os.environ.get('EVIDENCE_COMPARE_SESSIONS','82,83').split(',')):
  root=Path('artifacts')/{82:'evidence_session82_seed_stability',83:'evidence_session83_joint_policy',84:'evidence_session84_nested_joint_policy',92:'evidence_session92_chip_signatures'}[session];q=base
  if session==82:
   for offset in [0,10000,20000]:
    q=q.join(pl.read_parquet(root/f'seed{offset}'/'background_oof.parquet').select('pair_id','hand_id',C('cat_and_joint').alias(f'seed{offset}')),on=['pair_id','hand_id'],validate='1:1')
   np.testing.assert_array_equal(q['seed0'].to_numpy(),q['pressure59'].to_numpy())
   q=q.with_columns(pl.mean_horizontal('seed0','seed10000','seed20000').alias('candidate'));diagnostics=['seed0','seed10000','seed20000']
  else:
   q=q.join(pl.read_parquet(root/'background_oof.parquet').select('pair_id','hand_id',C('cat_and_joint').alias('candidate')),on=['pair_id','hand_id'],validate='1:1');diagnostics=[]
  q=q.with_columns(((C('candidate')+C('full'))*.5).alias('candidate_r33_recipe'));q.write_parquet(root/'combined_oof.parquet');names=['r33','pressure59',*diagnostics,'candidate','candidate_r33_recipe'];r,_=compare(root/'combined_oof.parquet',names,f'session{session}');pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={}
  for n in names:
   delta=pool[n].to_numpy()-pool['r33'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);out[n]={'MAP':r[n].mean(),'gain_vs_r33':r[n].mean()-r['r33'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows()),'better':int((r[n]>r['r33']+1e-12).sum()),'worse':int((r[n]<r['r33']-1e-12).sum())}
  (root/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
