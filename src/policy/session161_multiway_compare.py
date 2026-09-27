import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from session152_generic_fallback import ap
ROOT=Path('artifacts/evidence_session161_multiway_compare');C=pl.col

def main():
    source=Path('artifacts/evidence_session160_generic_multiway_control');v=json.load(open(source/'verification.json'))
    assert v['models']==8 and v['scope']=='all-family control only' and v['final_prediction_error']==0
    d=pl.read_parquet(source/'all/oof.parquet')
    rows=[dict(pair_id=pid,multiway=ap(g)) for (pid,),g in d.group_by('pair_id')]
    q=pl.read_csv('artifacts/evidence_session155_response_compare/pair_comparison.csv').join(pl.DataFrame(rows),on='pair_id',validate='1:1').sort('pair_id');assert len(q)==372
    ROOT.mkdir(exist_ok=True);q.write_csv(ROOT/'pair_comparison.csv');names=['cold','warm','response','multiway','r33']
    metrics={n:dict(MAP=q[n].mean(),families=dict(q.group_by('family').agg(C(n).mean()).iter_rows()),folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list()) for n in names}
    differences={}
    for fam in ['all','directed_transfer','soft_play','coordinated_isolation']:
        z=q if fam=='all' else q.filter(C('family')==fam);pool=z.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id')
        ix=np.random.default_rng(16101).integers(0,len(pool),(5000,len(pool)))
        for baseline in ['cold','warm','response','r33']:
            dd=pool['multiway'].to_numpy()-pool[baseline].to_numpy();boot=dd[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
            differences[fam+'__'+baseline]=dict(delta=z['multiway'].mean()-z[baseline].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist())
    out=dict(metrics=metrics,differences=differences,limitations='Fixed all-family control only. Cold is the matched feature intervention. Reused public labels; no withheld-family endpoint, fitted ensemble, or leaderboard claim.')
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
