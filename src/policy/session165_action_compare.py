import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from session152_generic_fallback import ap
ROOT=Path('artifacts/evidence_session165_action_compare');C=pl.col

def main():
    source=Path('artifacts/evidence_session164_generic_action_em');v=json.load(open(source/'verification.json'))
    assert v['models']==8 and v['scope']=='all-family control only' and v['final_prediction_error']==0
    d=pl.read_parquet(source/'oof.parquet');rows=[dict(pair_id=pid,action_em=ap(g)) for (pid,),g in d.group_by('pair_id')]
    q=pl.read_csv('artifacts/evidence_session155_response_compare/pair_comparison.csv').join(pl.DataFrame(rows),on='pair_id',validate='1:1').sort('pair_id');assert len(q)==372
    names=['cold','warm','response','action_em','r33'];mw=Path('artifacts/evidence_session161_multiway_compare/pair_comparison.csv')
    if mw.exists():q=q.join(pl.read_csv(mw).select('pair_id','multiway'),on='pair_id',validate='1:1');names.append('multiway')
    ROOT.mkdir(exist_ok=True);q.write_csv(ROOT/'pair_comparison.csv')
    metrics={n:dict(MAP=q[n].mean(),families=dict(q.group_by('family').agg(C(n).mean()).iter_rows()),folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list()) for n in names}
    differences={}
    for fam in ['all','directed_transfer','soft_play','coordinated_isolation']:
        z=q if fam=='all' else q.filter(C('family')==fam);pool=z.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id')
        ix=np.random.default_rng(16501).integers(0,len(pool),(5000,len(pool)))
        for baseline in [n for n in names if n!='action_em']:
            dd=pool['action_em'].to_numpy()-pool[baseline].to_numpy();boot=dd[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
            differences[fam+'__'+baseline]=dict(delta=z['action_em'].mean()-z[baseline].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist())
    report=dict(metrics=metrics,differences=differences,multiway_comparison_pending=not mw.exists(),limitations='Architectural alternative: coherent action inputs, hierarchical event probabilities and action-distributed pair weights differ from hand-summary controls. Fixed all-family folds only; no withheld-family result, fitted blend, submission or leaderboard estimate.')
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
