import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from session152_generic_fallback import ap
ROOT=Path('artifacts/evidence_session173_negative_action_compare');C=pl.col

def main():
    source=Path('artifacts/evidence_session172_negative_action_em');v=json.load(open(source/'verification.json'))
    assert v['models']==24 and v['final_prediction_error']==0 and not v['unknown_pairs_used']
    d=pl.read_parquet(source/'oof.parquet');rows=[dict(pair_id=pid,negative_action=ap(g)) for (pid,),g in d.group_by('pair_id')]
    q=pl.read_csv('artifacts/evidence_session168_extended_action_compare/pair_comparison.csv').join(pl.DataFrame(rows),on='pair_id',validate='1:1').sort('pair_id');assert len(q)==372
    names=['cold','action_em','action_em6','negative_action','r33'];ROOT.mkdir(exist_ok=True);q.write_csv(ROOT/'pair_comparison.csv')
    metrics={n:dict(MAP=q[n].mean(),families=dict(q.group_by('family').agg(C(n).mean()).iter_rows()),folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list()) for n in names};diff={}
    for family in ['all','directed_transfer','soft_play','coordinated_isolation']:
        z=q if family=='all' else q.filter(C('family')==family);p=z.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id')
        ix=np.random.default_rng(17301).integers(0,len(p),(5000,len(p)))
        for baseline in [n for n in names if n!='negative_action']:
            delta=p['negative_action'].to_numpy()-p[baseline].to_numpy();boot=delta[ix].sum(1)/p['n'].to_numpy()[ix].sum(1)
            diff[family+'__'+baseline]=dict(delta=z['negative_action'].mean()-z[baseline].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist())
    out=dict(metrics=metrics,differences=diff,limitations='Known-family control only; public pools repeatedly reused. No transfer claim, candidate or leaderboard improvement inferred.')
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
