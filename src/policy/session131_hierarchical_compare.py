import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
import numpy as np,polars as pl
from session130_hierarchical_list import ROOT,CONFIG
import session12_compare as comparator

def main():
    oof=pl.read_parquet(ROOT/'oof.parquet');assert oof.height==2*45129
    d=oof.filter(pl.col('kind')=='point').select('pair_id','hand_id',pl.col('score').alias('point'))
    d=d.join(oof.filter(pl.col('kind')=='random_rate').select('pair_id','hand_id',pl.col('score').alias('random_rate')),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',pl.col('equal').alias('r33'),'pressure59','full'),on=['pair_id','hand_id'],validate='1:1')
    d=d.with_columns(*[(.5*pl.col('pressure59')+.5*pl.col(n)).alias(n+'_r33_recipe') for n in ['point','random_rate']])
    names=['r33','full','point','random_rate','point_r33_recipe','random_rate_r33_recipe']
    d.write_parquet(ROOT/'comparison_input.parquet');comparator.ROOT=ROOT
    pairs,_=comparator.compare(ROOT/'comparison_input.parquet',names,'routed')
    assert abs(pairs['r33'].mean()-.7973334826762246)<1e-12
    pool=pairs.group_by('table_id').agg(pl.col(names).sum(),pl.len().alias('n')).sort('table_id')
    ix=np.random.default_rng(13100).integers(0,len(pool),(5000,len(pool)));out={}
    for n in names:
        delta=pool[n].to_numpy()-pool['r33'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
        out[n]=dict(MAP=pairs[n].mean(),gain_vs_r33=pairs[n].mean()-pairs['r33'].mean(),CI95=np.quantile(boot,[.025,.975]).tolist(),
                    folds=pairs.group_by('fold').agg(pl.col(n).mean()).sort('fold')[n].to_list(),
                    families=dict(pairs.group_by('family').agg(pl.col(n).mean()).iter_rows()))
    (ROOT/'r33_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
