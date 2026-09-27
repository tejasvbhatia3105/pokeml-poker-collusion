\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap

root = Path('artifacts/policy')
dest = root / 'monotone_novelty'
dest.mkdir(exist_ok=True)
labs = pl.read_csv('data/development_labels.csv').select('pair_id', 'label', 'behavior_family')
d = pl.scan_parquet(str(root / 'pair_features/*.parquet')).filter(pl.col('phase') == 'development').join(labs.lazy(), on='pair_id').collect().sort('pair_id')
cols = [c for c in json.loads((root / 'residual_columns.json').read_text()) if c.endswith(('_z', '_w10_max', '_w10_min'))]
X = d.select(cols).to_numpy()
y, b, g = (d[c].to_numpy() for c in ['label', 'behavior_family', 'table_id'])
folds = json.loads(Path('artifacts/folds.json').read_text())
previous = pd.read_csv(root / 'open_set_floor/oof.csv')
reports, outputs = [], []
t = time.time()
for family in ['all_known', 'directed_transfer', 'soft_play', 'coordinated_isolation']:
    preds = {kind: np.zeros(len(d)) for kind in ['unconstrained', 'monotone']}
    for f in folds:
        va = np.isin(g, f['valid_tables'])
        tr = ~va & ((b != family) if family != 'all_known' else True)
        normal = tr & (y == 0)
        med = np.median(X[normal], axis=0)
        scale = np.maximum(np.quantile(X[normal], .9, axis=0) - np.quantile(X[normal], .1, axis=0), .01)
        z = np.log1p(np.abs((X - med) / scale))
        for kind in preds:
            m = CatBoostClassifier(iterations=600, depth=4, learning_rate=.045,
                loss_function='Logloss', l2_leaf_reg=10, verbose=False,
                thread_count=4, random_seed=318 + f['fold'], allow_writing_files=False,
                monotone_constraints=[1 if kind == 'monotone' else 0] * len(cols))
            m.fit(z[tr], y[tr])
            preds[kind][va] = m.predict_proba(z[va])[:, 1]
            m.save_model(str(dest / f'{family}_{kind}_fold{f["fold"]}.cbm'))
        if family == 'all_known':
            np.savez(dest / f'scale_fold{f["fold"]}.npz', median=med, scale=scale, columns=np.array(cols))
        print(family, f['fold'], round(time.time() - t, 1), flush=True)
    base = previous[previous.withheld_family == family].set_index('pair_id').loc[d['pair_id'].to_list()]
    keep = np.ones(len(d), bool) if family == 'all_known' else (y == 0) | (b == family)
    for kind, pred in preds.items():
        for alpha in [0, .25, .5, .75, 1]:
            p = np.maximum(base.base_risk.to_numpy(), alpha * pred)
            reports.append(dict(withheld_family=family, method=kind, floor=alpha,
                weighted_AP=ap(y[keep], p[keep], sample_weight=np.where(y[keep] > 0, 1, 50))))
        reports.append(dict(withheld_family=family, method=kind, floor='standalone',
            weighted_AP=ap(y[keep], pred[keep], sample_weight=np.where(y[keep] > 0, 1, 50))))
    outputs.append(pd.DataFrame(dict(pair_id=d['pair_id'].to_list(), withheld_family=family,
        truth=y, behavior=b, **preds)))
    pd.concat(outputs).to_csv(dest / 'oof.csv', index=False)
    (dest / 'metrics.json').write_text(json.dumps(reports, indent=2))
    print('RESULT', family, [r for r in reports if r['withheld_family'] == family], flush=True)
