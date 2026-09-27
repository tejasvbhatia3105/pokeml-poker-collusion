\
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
dest = root / 'sparse_episodes'
dest.mkdir(exist_ok=True)
cols = json.loads((root / 'residual_columns.json').read_text())
folds = json.loads((root / 'table_folds.json').read_text())
labs = pl.read_csv('data/development_labels.csv').select('pair_id', 'label', 'behavior_family')
base = pl.read_parquet(root / 'exposure_full.parquet').sort('pair_id')
h = pl.scan_parquet(str(root / 'hand_features/*.parquet')).filter(pl.col('phase') == 'development').join(labs.lazy(), on='pair_id').collect().sort('pair_id', 'time_index')
channels = [c[:-2] for c in h.columns if c.endswith('_r')]
rawcols = [n + s for s in ['_r', '_v'] for n in channels] + ['surprise_max']
nc = len(channels)
def aggregate(a):
    r, v, surprise = a[:, :nc], a[:, nc:2*nc], a[:, -1]
    hz = r / np.sqrt(v + 1)
    out = {'policy_n_hands': len(a), 'policy_surprise_mean': surprise.mean(),
           'policy_surprise_top5': np.sort(surprise)[-5:].mean()}
    for j, n in enumerate(channels):
        out.update({n+'_z': r[:,j].sum()/np.sqrt(v[:,j].sum()+1),
            n+'_mean': hz[:,j].mean(), n+'_std': hz[:,j].std(ddof=1) if len(a)>1 else 0,
            n+'_top5': np.sort(hz[:,j])[-5:].mean(),
            n+'_bottom5': np.sort(-hz[:,j])[-5:].mean()})
    cr = np.vstack([np.zeros(nc), np.cumsum(r, axis=0)])
    cv = np.vstack([np.zeros(nc), np.cumsum(v, axis=0)])
    ix = np.arange(1, len(a)+1)
    for w in [5, 10, 20]:
        z = (cr[ix]-cr[np.maximum(ix-w,0)]) / np.sqrt(cv[ix]-cv[np.maximum(ix-w,0)]+1)
        for j, n in enumerate(channels):
            out[n+f'_w{w}_max'] = z[:,j].max()
            out[n+f'_w{w}_min'] = z[:,j].min()
    return np.array([out[c] for c in cols])

names = ['none', 'directed_transfer', 'soft_play', 'coordinated_isolation']
records = {row['pair_id']: row for row in base.iter_rows(named=True)}
histories = {key[0]: d for key, d in h.partition_by('pair_id', as_dict=True).items()}
evidence = pl.read_csv('data/development_evidence.csv')
ev_ids = {key[0]: set(d['hand_id']) for key, d in evidence.partition_by('pair_id', as_dict=True).items()}
rng = np.random.default_rng(4938)
normal_ids = [p for p, row in records.items() if row['label'] == 0]
                                                                               
errors = []
for pid in normal_ids[:20]:
    rec = aggregate(histories[pid].select(rawcols).to_numpy().astype(float))
    errors.append(np.max(np.abs(rec - np.array([records[pid][c] for c in cols]))))
assert max(errors) < 1e-4, max(errors)
print('aggregation_max_error', max(errors), flush=True)
synthetic, meta = [], []
t = time.time()
for pid in sorted(ev_ids):
    row = records[pid]
    fold = folds[row['table_id']]
    donors = histories[pid].filter(pl.col('hand_id').is_in(ev_ids[pid])).select(rawcols).to_numpy().astype(float)
    samepool = [p for p in normal_ids if records[p]['table_id'] == row['table_id']]
    backgrounds = samepool or [p for p in normal_ids if folds[records[p]['table_id']] == fold]
    for k in [1, 3, 5]:
        for rep in range(2):
            bg = rng.choice(backgrounds)
            a = histories[bg].select(rawcols).to_numpy().astype(float)
                                                                             
            n = max(10, int(len(a)*2/3))
            start = rng.integers(max(1, len(a)-n+1))
            a = a[start:start+n].copy()
            count = min(k, len(donors), len(a))
            block = min(20, len(a))
            pos = rng.integers(max(1, len(a)-block+1))
            slots = np.sort(rng.choice(np.arange(pos,pos+block), count, replace=False))
            a[slots] = donors[np.sort(rng.choice(len(donors), count, replace=False))]
            synthetic.append(aggregate(a))
            meta.append(dict(pair_id=pid, background_pair_id=bg, fold=fold,
                truth=names.index(row['behavior_family']), planted_count=count))
print('synthetic_rows', len(synthetic), 'seconds', round(time.time()-t,1), flush=True)
SX = np.array(synthetic)
sm = pd.DataFrame(meta)
sm.to_csv(dest/'augmentation_manifest.csv', index=False)
np.save(dest/'synthetic_features.npy', SX)
parts = []
for window in ['full', 'first_2000', 'last_2000']:
    d = pl.read_parquet(root/f'exposure_{window}.parquet').join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(pl.col('eligible')).with_columns(pl.lit(window).alias('window'))
    parts.append(d)
d = pl.concat(parts)
X = d.select(cols).to_numpy()
y = np.array([names.index(b) for b in d['behavior_family']])
groups = np.array([folds[t] for t in d['table_id']])
orig = d['window'].to_numpy() == 'full'
basep = np.zeros((len(d), 4))
for f in range(4):
    va = groups == f
    m = CatBoostClassifier(); m.load_model(str(root/f'residual_fold{f}.cbm'))
    basep[va] = m.predict_proba(X[va])
reports = []
for aug_weight in [.25, 1.0]:
    pred = np.zeros((len(d), 4))
    for f in range(4):
        va = groups == f
        tr = ~va & orig
        st = sm.fold.to_numpy() != f
        TX = np.concatenate([X[tr], SX[st]])
        ty = np.concatenate([y[tr], sm.truth.to_numpy()[st]])
                                                                         
        weights = np.concatenate([np.ones(tr.sum()), np.full(st.sum(), aug_weight/6)])
        m = CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,
            loss_function='MultiClass',l2_leaf_reg=10,thread_count=4,
            random_seed=991+f,verbose=False,allow_writing_files=False)
        m.fit(TX,ty,sample_weight=weights)
        pred[va] = m.predict_proba(X[va])
        m.save_model(str(dest/f'weight{aug_weight}_fold{f}.cbm'))
        print('fit',aug_weight,f,'seconds',round(time.time()-t,1),flush=True)
    pd.DataFrame(pred,columns=names).assign(pair_id=d['pair_id'].to_list(),window=d['window'].to_list(),truth=y).to_csv(dest/f'oof_weight{aug_weight}.csv',index=False)
    for blend in [0,.5,1]:
        pp = (1-blend)*basep+blend*pred
        for window in ['full','first_2000','last_2000']:
            keep = d['window'].to_numpy()==window
            yy = y[keep]; risk=1-pp[keep,0]; behavior=pp[keep,1:].argmax(1)+1
            w=np.where(yy>0,1,50)
            reports.append(dict(augmentation_weight=aug_weight,blend=blend,window=window,
                AP=ap(yy>0,risk),weighted_AP=ap(yy>0,risk,sample_weight=w),
                weighted_behavior_AP=float(np.mean([ap(yy==k,risk*(behavior==k),sample_weight=w) for k in [1,2,3]]))))
    (dest/'metrics.json').write_text(json.dumps(reports,indent=2))
    print('RESULT',aug_weight,[r for r in reports if r['augmentation_weight']==aug_weight],flush=True)
