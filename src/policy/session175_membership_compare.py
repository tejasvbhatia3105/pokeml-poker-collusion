import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from session152_generic_fallback import ap
import session174_direct_list_membership as direct
ROOT=Path('artifacts/evidence_session175_membership_compare');C=pl.col

def main():
    assert json.load(open(direct.ROOT/'verification.json'))['models']==16
    rows=[]
    for held in ['all']+direct.s.FAMILIES:
        a=pl.read_parquet(direct.ROOT/held/'oof.parquet');b=pl.read_parquet(direct.s.ROOT/held/'oof.parquet')
        assert a.select('pair_id','hand_id').sort('pair_id','hand_id').equals(b.select('pair_id','hand_id').sort('pair_id','hand_id'))
        other={pid:ap(g) for (pid,),g in b.group_by('pair_id')}
        for (pid,),g in a.group_by('pair_id'):
            rows.append(dict(held=held,pair_id=pid,table_id=g['table_id'][0],fold=g['fold'][0],family=g['behavior_family'][0],direct=ap(g),cold=other[pid]))
    r=pl.DataFrame(rows);ROOT.mkdir(exist_ok=True);r.write_csv(ROOT/'pair_comparison.csv');result={}
    for held in ['all','withheld']+direct.s.FAMILIES:
        z=r.filter(C('held')!='all') if held=='withheld' else r.filter(C('held')==held)
        p=z.group_by('table_id').agg(C('direct').sum(),C('cold').sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(17501).integers(0,len(p),(5000,len(p)))
        delta=p['direct'].to_numpy()-p['cold'].to_numpy();boot=delta[ix].sum(1)/p['n'].to_numpy()[ix].sum(1)
        result[held]=dict(direct=z['direct'].mean(),cold=z['cold'].mean(),delta=z['direct'].mean()-z['cold'].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist(),folds=z.group_by('fold').agg(C('direct').mean(),C('cold').mean()).sort('fold').to_dicts())
    out=dict(results=result,limitations='Same publicly reused pool splits. Direct predictions classify capped-list membership, not event absence. Generic transfer test; no unknown-family truth, candidate or leaderboard gain.')
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
