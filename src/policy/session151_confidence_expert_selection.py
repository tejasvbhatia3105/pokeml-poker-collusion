\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT=Path('artifacts/evidence_session151_confidence_expert_selection')
C=pl.col

def main():
    ROOT.mkdir(exist_ok=True)
    support=pl.read_parquet('artifacts/evidence_session144_expert_support/expert_support.parquet')
    metrics=pl.read_csv('artifacts/evidence_session142_expert_transfer/pair_comparison.csv')
    rows=[]
    for r in metrics.iter_rows(named=True):
        available=support.filter((C('pair_id')==r['pair_id'])&(C('expert')!=r['family']))
        assert len(available)==2
        expert=available.sort('raw_top5_confidence','expert',descending=[True,False])['expert'][0]
        assert expert!=r['family']
        rows.append(dict(pair_id=r['pair_id'],table_id=r['table_id'],fold=r['fold'],family=r['family'],chosen_expert=expert,
            confidence_selected=r['conditioned_'+expert],retained_mean=r['conditioned_other_mean'],
            best_single_oracle=r['conditioned_other_best_single_oracle']))
    q=pl.DataFrame(rows).sort('pair_id');assert len(q)==372
    q.write_csv(ROOT/'pair_comparison.csv')
    names=['confidence_selected','retained_mean','best_single_oracle']
    report={n:dict(MAP=q[n].mean(),families=dict(q.group_by('family').agg(C(n).mean()).iter_rows()),
        folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list()) for n in names}
    pool=q.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id')
    ix=np.random.default_rng(15101).integers(0,len(pool),(5000,len(pool)))
    dd=pool['confidence_selected'].to_numpy()-pool['retained_mean'].to_numpy()
    boot=dd[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
    out=dict(method=__doc__,metrics=report,delta_to_mean=q['confidence_selected'].mean()-q['retained_mean'].mean(),
        pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist(),
        limitations='Exploratory fixed rule, after prior support analysis. Experts exclude the query pool fold and unavailable family labels. Not a full R33 leave-family-out model, hidden-family estimate, generic fallback rule, or submission.')
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
