\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap

root = Path('artifacts/policy')
dest = root / 'value_unseen'
dest.mkdir(exist_ok=True)
labs = pl.read_csv('data/development_labels.csv').select('pair_id', 'label', 'behavior_family')
core = pl.scan_parquet(str(root / 'pair_features/*.parquet')).filter(pl.col('phase') == 'development').join(labs.lazy(), on='pair_id').collect()
values = pl.scan_parquet(str(root / 'value_pairs/*.parquet')).filter(pl.col('phase') == 'development').collect()
cols = json.loads((root / 'value_model/columns.json').read_text())
vc = [c for c in cols if c.startswith('value_')]
d = core.join(values.select('pair_id', *vc), on='pair_id', how='left').sort('pair_id')
assert d.height == labs.height and d.select(cols).null_count().to_numpy().sum() == 0
X = d.select(cols).to_numpy()
assert np.isfinite(X).all()
y = d['label'].to_numpy()
b = d['behavior_family'].to_numpy()
g = d['table_id'].to_numpy()
folds = json.loads(Path('artifacts/folds.json').read_text())
previous = pd.read_csv(root / 'open_set_floor/oof.csv')
reports, outputs = [], []
t = time.time()
for family in ['directed_transfer', 'soft_play', 'coordinated_isolation']:
    pred = np.zeros(len(d))
    for f in folds:
        va = np.isin(g, f['valid_tables'])
        tr = ~va & (b != family)
        m = CatBoostClassifier(iterations=600, depth=4, learning_rate=.045,
            loss_function='Logloss', l2_leaf_reg=10, verbose=False,
            thread_count=4, random_seed=318 + f['fold'], allow_writing_files=False)
        m.fit(X[tr], y[tr])
        pred[va] = m.predict_proba(X[va])[:, 1]
        m.save_model(str(dest / f'{family}_fold{f["fold"]}.cbm'))
        print(family, 'fold', f['fold'], 'seconds', round(time.time() - t, 1), flush=True)
    base = previous[previous.withheld_family == family].set_index('pair_id').loc[d['pair_id'].to_list()]
    keep = (y == 0) | (b == family)
    for alpha in [0, .5, 1]:
        mix = (1 - alpha) * base.base_risk.to_numpy() + alpha * pred
        for floor in [0, .25, 1]:
            p = np.maximum(mix, floor * base.novelty_probability.to_numpy())
            reports.append(dict(withheld_family=family, value_weight=alpha,
                novelty_floor=floor, weighted_AP=ap(y[keep], p[keep],
                sample_weight=np.where(y[keep] > 0, 1, 50))))
    outputs.append(pd.DataFrame(dict(pair_id=d['pair_id'].to_list(),
        withheld_family=family, truth=y, behavior=b, value_risk=pred)))
    pd.concat(outputs).to_csv(dest / 'oof.csv', index=False)
    (dest / 'metrics.json').write_text(json.dumps(reports, indent=2))
    print('RESULT', family, [r for r in reports if r['withheld_family'] == family], flush=True)
