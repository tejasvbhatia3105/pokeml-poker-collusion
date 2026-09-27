import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from session152_generic_fallback import ap

ROOT=Path('artifacts/evidence_session155_response_compare');C=pl.col

def main():
    source=Path('artifacts/evidence_session154_response_control')
    v=json.load(open(source/'verification.json'))
    assert v['models']==8 and v['scope']=='all-family control only' and v['final_prediction_error']==0
    d=pl.read_parquet(source/'all/oof.parquet');rows=[]
    for (pid,),g in d.group_by('pair_id'):
        rows.append(dict(pair_id=pid,response=ap(g),fold=g['fold'][0],family=g['behavior_family'][0],table_id=g['table_id'][0]))
    q=pl.DataFrame(rows)
    controls=pl.read_csv('artifacts/evidence_session146_warm_generic/control_only_pair_comparison.csv')
    q=q.join(controls.select('pair_id','cold','warm'),on='pair_id',validate='1:1')
    reference=pl.read_csv('artifacts/evidence_session149_persistent_donor/retrieval_comparison.csv')
    q=q.join(reference.select('pair_id','r33'),on='pair_id',validate='1:1').sort('pair_id');assert len(q)==372
    ROOT.mkdir(exist_ok=True);q.write_csv(ROOT/'pair_comparison.csv')
    names=['cold','warm','response','r33']
    metrics={n:dict(MAP=q[n].mean(),families=dict(q.group_by('family').agg(C(n).mean()).iter_rows()),folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list()) for n in names}
    differences={}
    for fam in ['all','directed_transfer','soft_play','coordinated_isolation']:
        z=q if fam=='all' else q.filter(C('family')==fam)
        pool=z.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id')
        ix=np.random.default_rng(15501).integers(0,len(pool),(5000,len(pool)))
        for baseline in ['cold','warm','r33']:
            dd=pool['response'].to_numpy()-pool[baseline].to_numpy();boot=dd[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
            differences[fam+'__'+baseline]=dict(delta=z['response'].mean()-z[baseline].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist())
    report=dict(metrics=metrics,differences=differences,limitations='Fixed all-family control on reused public labels. Cold is the matched feature-intervention comparator. Warm differs in initialization as well as features. No unseen-family endpoint, inferred leaderboard gain, blend or submission.')
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
