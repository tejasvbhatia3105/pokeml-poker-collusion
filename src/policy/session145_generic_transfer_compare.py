import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
from pathlib import Path
import numpy as np,polars as pl

ROOT=Path('artifacts/evidence_session145_generic_transfer_compare');C=pl.col

def main():
    ROOT.mkdir(exist_ok=True)
    verify=json.load(open('artifacts/evidence_session123_family_holdout/verification.json'))
    assert verify['models']==32 and verify['family_exclusions_tested'] and verify['final_prediction_error']==0
    generic=pl.read_csv('artifacts/evidence_session123_family_holdout/pair_comparison.csv')
    expert=pl.read_csv('artifacts/evidence_session142_expert_transfer/pair_comparison.csv')
    q=generic.join(expert.select('pair_id',C('family').alias('expert_family'),C('fold').alias('expert_fold'),'raw_native','raw_other_mean','conditioned_native','conditioned_other_mean','conditioned_other_max','conditioned_other_best_single_oracle'),on='pair_id',validate='1:1').sort('pair_id')
    assert len(q)==372 and (q['family']==q['expert_family']).all() and (q['fold']==q['expert_fold']).all()
    q=q.rename({'all_families':'generic_all','withheld_family':'generic_withheld'})
    selector=pl.read_csv('artifacts/evidence_session151_confidence_expert_selection/pair_comparison.csv')
    q=q.join(selector.select('pair_id','confidence_selected'),on='pair_id',validate='1:1')
    q.write_csv(ROOT/'pair_comparison.csv')
    names=['generic_all','generic_withheld','raw_native','raw_other_mean','conditioned_native','conditioned_other_mean','conditioned_other_max','conditioned_other_best_single_oracle','confidence_selected','r33_known_family_reference']
    report={n:dict(MAP=q[n].mean(),families=dict(q.group_by('family').agg(C(n).mean()).iter_rows()),folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list()) for n in names}
    differences={}
    for family in ['all','directed_transfer','soft_play','coordinated_isolation']:
        z=q if family=='all' else q.filter(C('family')==family)
        pool=z.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(14501).integers(0,len(pool),(5000,len(pool)))
        for control in ['raw_other_mean','conditioned_other_mean','conditioned_other_best_single_oracle','confidence_selected','generic_all']:
            delta=pool['generic_withheld'].to_numpy()-pool[control].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);differences[family+'__'+control]=dict(delta=z['generic_withheld'].mean()-z[control].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist(),better=int((z['generic_withheld']>z[control]+1e-12).sum()),worse=int((z['generic_withheld']<z[control]-1e-12).sum()))
    out=dict(metrics=report,differences=differences,limitations='Conditional retrieval on reused public families; no detector/gate/evaluation inference or candidate. Generic withheld vs retained experts is the relevant new comparison. Native experts and generic all-family models are contextual controls, not withheld-family baselines. Best-single expert oracle is not deployable. No estimate of hidden-family prevalence or true leaderboard improvement.')
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
