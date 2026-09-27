import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from session152_generic_fallback import ap
ROOT=Path('artifacts/evidence_session184_depthwise_compare');C=pl.col

def main():
    source=Path('artifacts/evidence_session183_depthwise_action_em');v=json.load(open(source/'verification.json'));g=json.load(open(source/'growth_policy_verification.json'))
    assert v['models']==len(g['records'])==24 and v['final_prediction_error']==0 and g['initial_targets_match167']
    d=pl.read_parquet(source/'oof.parquet');rows=[dict(pair_id=pid,depthwise=ap(z)) for (pid,),z in d.group_by('pair_id')]
    q=pl.read_csv('artifacts/evidence_session168_extended_action_compare/pair_comparison.csv').join(pl.DataFrame(rows),on='pair_id',validate='1:1').sort('pair_id');assert len(q)==372
    names=['cold','action_em6','depthwise','r33'];ROOT.mkdir(exist_ok=True);q.write_csv(ROOT/'pair_comparison.csv');metrics={n:dict(MAP=q[n].mean(),families=dict(q.group_by('family').agg(C(n).mean()).iter_rows()),folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list()) for n in names};diff={}
    for family in ['all','directed_transfer','soft_play','coordinated_isolation']:
        z=q if family=='all' else q.filter(C('family')==family);p=z.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(18401).integers(0,len(p),(5000,len(p)))
        for baseline in ['cold','action_em6','r33']:
            delta=p['depthwise'].to_numpy()-p[baseline].to_numpy();boot=delta[ix].sum(1)/p['n'].to_numpy()[ix].sum(1)
            diff[family+'__'+baseline]=dict(delta=z['depthwise'].mean()-z[baseline].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist())
    out=dict(metrics=metrics,differences=diff,limitations='All-family control only, on reused public pools. No transfer, candidate, or leaderboard improvement inferred. Only growth policy and its downstream EM targets differ from167.')
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
