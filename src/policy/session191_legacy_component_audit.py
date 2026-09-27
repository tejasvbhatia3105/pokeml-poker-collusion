\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import json
from pathlib import Path
import numpy as np
import polars as pl
from session189_pair_event_prototypes import data

ROOT = Path('artifacts/evidence_session191_legacy_component_audit')
C = pl.col

def main():
    d = data()
    keys = ['pair_id', 'hand_id']
    current = pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet')
    base = pl.concat([
        pl.read_parquet(f'artifacts/evidence_session55_current_nested/nested_outer{f}.parquet')
        .filter(C('fold') == f).select(*keys, 'base') for f in range(4)
    ])
    fallback = pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet')
    risk = pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window') == 'full')
    q = d.join(current.select(*keys, 'equal'), on=keys, validate='1:1')
    q = q.join(base, on=keys, validate='1:1').join(fallback.select(*keys, 'conditional_family'), on=keys, validate='1:1')
    q = q.join(risk.select('pair_id', 'risk_score'), on='pair_id', validate='m:1')
    q = q.with_columns(((C('equal') - .25*C('base'))/.75).alias('without_legacy_base'))
    assert q['without_legacy_base'].min() >= -1e-6 and q['without_legacy_base'].max() <= 1+1e-6
    q = q.with_columns(C('equal').alias('r33')).with_columns(*[
        pl.when(C('risk_score') < .05).then(C('conditional_family')).otherwise(C(n)).alias(n)
        for n in ['r33', 'without_legacy_base']
    ])
    rows = []
    for (pid,), g in q.group_by('pair_id'):
        r = dict(pair_id=pid, table_id=g['table_id'][0], fold=g['fold'][0])
        for n in ['r33', 'without_legacy_base']:
            y = g.sort([n, 'hand_id'], descending=[True, False])['evidence'].to_numpy()[:5]
            r[n] = float((y*np.cumsum(y)/np.arange(1, len(y)+1)).sum()/min(5, g['evidence'].sum()))
        rows.append(r)
    p = pl.DataFrame(rows).sort('pair_id')
    assert abs(p['r33'].mean()-.7973334826762246) < 1e-12
    assert abs(p['without_legacy_base'].mean()-.7931757765830346) < 1e-12
    ROOT.mkdir(exist_ok=True)
    p.write_csv(ROOT/'replayed_pairs.csv')
    report = dict(method=__doc__, baseline=p['r33'].mean(), without_legacy_base=p['without_legacy_base'].mean(), exact_saved_MAP_replay=True)
    (ROOT/'replay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
