import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
import numpy as np,polars as pl
import session12_compare as s
from session149_persistent_donor_inference import ROOT

def main():
    s.ROOT=ROOT;q,_=s.compare(ROOT/'oof.parquet',['r33','donor_integrated'],'retrieval')
    assert abs(q['r33'].mean()-.7973334826762246)<1e-12
    pool=q.group_by('table_id').agg(pl.col('r33','donor_integrated').sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(15001).integers(0,len(pool),(5000,len(pool)));dd=pool['donor_integrated'].to_numpy()-pool['r33'].to_numpy();boot=dd[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
    report=dict(r33_MAP=q['r33'].mean(),new_MAP=q['donor_integrated'].mean(),delta=q['donor_integrated'].mean()-q['r33'].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist(),folds=q.group_by('fold').agg(pl.col('r33','donor_integrated').mean()).sort('fold').to_dicts(),families=q.group_by('family').agg(pl.col('r33','donor_integrated').mean()).to_dicts(),better=int((q['donor_integrated']>q['r33']+1e-12).sum()),worse=int((q['donor_integrated']<q['r33']-1e-12).sum()))
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
