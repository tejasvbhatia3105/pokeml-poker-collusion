import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
import numpy as np,polars as pl
import session12_compare as comparator
from session138_private_policy_heads import ROOT,ARMS

def main():
    q=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',pl.col('equal').alias('r33'),'pressure59','full')
    for arm in ARMS:
        z=pl.read_parquet(ROOT/arm/'background_oof.parquet').select('pair_id','hand_id',pl.col('cat_and_joint').alias(arm))
        q=q.join(z,on=['pair_id','hand_id'],validate='1:1').with_columns(((pl.col(arm)+pl.col('full'))*.5).alias(arm+'_r33_recipe'))
    q.write_parquet(ROOT/'combined_oof.parquet');names=['r33','pressure59',*ARMS,*[k+'_r33_recipe' for k in ARMS]]
    comparator.ROOT=ROOT;r,_=comparator.compare(ROOT/'combined_oof.parquet',names,'retrieval')
    assert abs(r['r33'].mean()-.7973334826762246)<1e-12
    pool=r.group_by('table_id').agg(pl.col(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));out={}
    for n in names:
        delta=pool[n].to_numpy()-pool['r33'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
        out[n]=dict(MAP=r[n].mean(),gain_vs_r33=r[n].mean()-r['r33'].mean(),fixed_prediction_pool_CI95=np.quantile(boot,[.025,.975]).tolist(),
                    folds=r.group_by('fold').agg(pl.col(n).mean()).sort('fold')[n].to_list(),families=dict(r.group_by('family').agg(pl.col(n).mean()).iter_rows()),
                    better=int((r[n]>r['r33']+1e-12).sum()),worse=int((r[n]<r['r33']-1e-12).sum()))
    a='public_private_r33_recipe';b='public_r33_recipe';delta=pool[a].to_numpy()-pool[b].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
    out['private_vs_public']=dict(gain=r[a].mean()-r[b].mean(),CI95=np.quantile(boot,[.025,.975]).tolist())
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
