\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from catboost import CatBoostClassifier

root = Path('artifacts/policy')
dest = root / 'anomaly_redundancy'
dest.mkdir(exist_ok=True)
params = np.load(root / 'novelty_parameters.npz')
nc = params['columns'].tolist()
groups = np.array([c.removesuffix('_z').removesuffix('_w10_max').removesuffix('_w10_min') for c in nc])
unique = np.unique(groups)
cols = json.loads((root / 'residual_columns.json').read_text())
folds = json.loads(Path('artifacts/folds.json').read_text())

def scores(z):
    grouped = np.stack([z[:, groups == g].max(axis=1) for g in unique], axis=1)
    grouped_mean = np.stack([z[:, groups == g].mean(axis=1) for g in unique], axis=1)
    return dict(original=np.sort(z, axis=1)[:, -3:].mean(1),
                distinct_max=np.sort(grouped, axis=1)[:, -3:].mean(1),
                distinct_mean=np.sort(grouped_mean, axis=1)[:, -3:].mean(1))

a = pl.scan_parquet(str(root / 'pair_features/*.parquet')).filter(pl.col('phase') == 'evaluation').select('pair_id', *nc).collect().sort('pair_id')
x = a.select(nc).to_numpy()
z = np.mean([np.log1p(abs((x - m) / s)) for m, s in zip(params['med'], params['scale'])], axis=0)
top = groups[np.argsort(z, axis=1)[:, -3:]]
counts = np.array([len(set(row)) for row in top])
frame = pd.DataFrame(dict(pair_id=a['pair_id'].to_list(), distinct_top3_channels=counts, top_channel=groups[np.argmax(z, axis=1)]))
v4 = pd.read_csv('artifacts/v4/submission.csv').set_index('pair_id')
v5 = pd.read_csv('artifacts/v5/submission.csv').set_index('pair_id')
assert v4.filter(like='evidence_').equals(v5.filter(like='evidence_'))
frame = frame.set_index('pair_id').loc[v4.index]
frame['v4_risk'] = v4.risk_score
frame['v5_risk'] = v5.risk_score
frame['v4_rank'] = v4.risk_score.rank(ascending=False, method='min')
frame['v5_rank'] = v5.risk_score.rank(ascending=False, method='min')
frame['rank_gain'] = frame.v4_rank - frame.v5_rank
frame.to_csv(dest / 'evaluation_rank_audit.csv')
audit = []
for k in [100, 500, 1000]:
    before = set(v4.sort_values('risk_score', ascending=False).head(k).index)
    after = set(v5.sort_values('risk_score', ascending=False).head(k).index)
    promoted = frame.loc[sorted(after - before)]
    audit.append(dict(top_k=k, new_pairs=len(promoted), repeated_channel_top3=int((promoted.distinct_top3_channels < 3).sum()), top_channels=promoted.top_channel.value_counts().to_dict(), median_original_risk=float(promoted.v4_risk.median())))
(dest / 'evaluation_rank_audit.json').write_text(json.dumps(audit, indent=2))
print('EVALUATION', audit, flush=True)

cross = pd.read_csv(root / 'crossed_validation/oof.csv')
cross = cross[cross.training == 'full_training']
outputs = []
reports = []
for window in ['full', 'first_2000', 'last_2000']:
    folder = root / 'pair_features' if window == 'full' else root / 'full_window_stress' / window / 'pair_features'
    a = pl.scan_parquet(str(folder / '*.parquet')).filter(pl.col('phase') == 'development').collect().sort('pair_id')
    parts = []
    for fold in folds:
        q = a.filter(pl.col('table_id').is_in(fold['valid_tables']))
        p = np.load(root / f'monotone_novelty/scale_fold{fold["fold"]}.npz')
        assert p['columns'].tolist() == nc
        z = np.log1p(abs((q.select(nc).to_numpy() - p['median']) / p['scale']))
        vals = scores(z)
        m = CatBoostClassifier()
        m.load_model(str(root / f'residual_fold{fold["fold"]}.cbm'))
        vals['v4_rarity'] = 1 - m.predict_proba(q.select(cols).to_numpy(), thread_count=4)[:, 0]
        parts.append(pd.DataFrame(dict(pair_id=q['pair_id'].to_list(), **{k: -np.log1p(-rankdata(v)/(len(v)+1)) for k, v in vals.items()})))
    pred = pd.concat(parts)
    d = cross[cross.window == window].merge(pred, on='pair_id', validate='many_to_one')
    for family, q in d.groupby('withheld_family'):
        q = q if family == 'all_known' else q[(q.truth == 0) | (q.behavior == family)]
        base = q.v4_rarity if family == 'all_known' else q.rarity
        for variant in ['base', 'original', 'distinct_max', 'distinct_mean']:
            score = base if variant == 'base' else np.maximum(base, .95*q[variant])
            reports.append(dict(window=window, family=family, variant=variant, weighted_AP=ap(q.truth, score, sample_weight=np.where(q.truth > 0, 1, 50))))
    outputs.append(d)
    print('DONE', window, flush=True)
pd.concat(outputs).to_csv(dest / 'oof.csv', index=False)
pd.DataFrame(reports).to_csv(dest / 'metrics.csv', index=False)
print(pd.DataFrame(reports).pivot(index=['window','family'],columns='variant',values='weighted_AP').to_string(), flush=True)
