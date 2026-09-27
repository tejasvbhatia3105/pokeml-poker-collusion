import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
from pathlib import Path
import numpy as np,polars as pl
ROOT=Path('artifacts/evidence_session147_warm_transfer_compare');C=pl.col

def main():
    ROOT.mkdir(exist_ok=True)
    for root in ['artifacts/evidence_session123_family_holdout','artifacts/evidence_session146_warm_generic']:
        v=json.load(open(Path(root)/'verification.json'))
        assert v['models']==32 and v['family_exclusions_tested'] and v['final_prediction_error']==0
    cold=pl.read_csv('artifacts/evidence_session145_generic_transfer_compare/pair_comparison.csv')
    warm=pl.read_csv('artifacts/evidence_session146_warm_generic/pair_comparison.csv').select('pair_id',C('family').alias('warm_family'),C('fold').alias('warm_fold'),C('all_families').alias('warm_all'),C('withheld_family').alias('warm_withheld'))
    q=cold.join(warm,on='pair_id',validate='1:1').sort('pair_id')
    assert len(q)==372 and (q['family']==q['warm_family']).all() and (q['fold']==q['warm_fold']).all()
    q=q.rename({'generic_all':'cold_all','generic_withheld':'cold_withheld'});q.write_csv(ROOT/'pair_comparison.csv')
    names=['cold_all','cold_withheld','warm_all','warm_withheld','conditioned_native','conditioned_other_mean','conditioned_other_best_single_oracle','confidence_selected','r33_known_family_reference']
    report={}
    for n in names:
        families=dict(q.group_by('family').agg(C(n).mean()).iter_rows());report[n]=dict(MAP=q[n].mean(),macro_family_MAP=float(np.mean(list(families.values()))),families=families,folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list())
    differences={}
    for family in ['all','directed_transfer','soft_play','coordinated_isolation']:
        z=q if family=='all' else q.filter(C('family')==family);pool=z.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(14701).integers(0,len(pool),(5000,len(pool)))
        for first,second in [('warm_withheld','cold_withheld'),('warm_withheld','conditioned_other_mean'),('warm_withheld','conditioned_other_best_single_oracle'),('warm_withheld','confidence_selected'),('warm_all','cold_all')]:
            dd=pool[first].to_numpy()-pool[second].to_numpy();boot=dd[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);differences[family+'__'+first+'__'+second]=dict(delta=z[first].mean()-z[second].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist(),better=int((z[first]>z[second]+1e-12).sum()),worse=int((z[first]<z[second]-1e-12).sum()))
    out=dict(metrics=report,differences=differences,limitations='Simulated conditional retrieval with known families withheld, not an estimate of the undisclosed mechanism or public leaderboard gain. No validated fallback gate, evaluation inference, or candidate. One predeclared initialization change; no fitted ensemble or per-family splice.')
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
