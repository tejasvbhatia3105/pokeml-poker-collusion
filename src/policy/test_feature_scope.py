import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'feature_scope';dest.mkdir(exist_ok=True)
core=json.loads((root/'residual_columns.json').read_text());parts=[]
for window in ['full','first_2000','last_2000']:
    d=pl.read_parquet(root/f'exposure_{window}.parquet').join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(pl.col('eligible')).with_columns(pl.lit(window).alias('window'));parts.append(d)
d=pl.concat(parts).with_columns((1/pl.len().over('pair_id')).alias('weight'))
names=['none','directed_transfer','soft_play','coordinated_isolation'];y=np.array([names.index(b) for b in d['behavior_family']]);g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());base=np.zeros((len(d),4));t=time.time()
for f in folds:
    va=np.isin(g,f['valid_tables']);m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));base[va]=m.predict_proba(d.filter(pl.Series(va)).select(core).to_numpy())
scopes={'local_only':[c for c in core if any(f'_w{w}_' in c for w in [5,10,20]) or c.endswith(('_top5','_bottom5'))],
    'partner_specific':[c for c in core if not c.startswith(('alive_','dealt_weak_agg','policy_'))]}
reports=[]
if os.environ.get('SCOPE'):
    scopes={k:v for k,v in scopes.items() if k==os.environ['SCOPE']}
    if (dest/'metrics.json').exists(): reports=[r for r in json.loads((dest/'metrics.json').read_text()) if r['scope'] not in scopes]
for scope,cols in scopes.items():
    X=d.select(cols).to_numpy();p=np.zeros_like(base)
    for f in folds:
        va=np.isin(g,f['valid_tables']);tr=~va
        m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=4,random_seed=991+f['fold'],verbose=False,allow_writing_files=False)
        m.fit(X[tr],y[tr],sample_weight=d['weight'].to_numpy()[tr]);p[va]=m.predict_proba(X[va]);m.save_model(str(dest/f'{scope}_fold{f["fold"]}.cbm'));print(scope,f['fold'],round(time.time()-t,1),flush=True)
    pd.DataFrame(p,columns=names).assign(pair_id=d['pair_id'].to_list(),window=d['window'].to_list(),truth=y).to_csv(dest/f'{scope}_oof.csv',index=False);(dest/f'{scope}_columns.json').write_text(json.dumps(cols))
    for alpha in [0,.5,1]:
        mix=alpha*p+(1-alpha)*base
        for window in ['full','first_2000','last_2000']:
            keep=d['window'].to_numpy()==window;yy=y[keep];risk=1-mix[keep,0];behavior=mix[keep,1:].argmax(1)+1;w=np.where(yy>0,1,50)
            reports.append(dict(scope=scope,weight=alpha,window=window,AP=ap(yy>0,risk),weighted_AP=ap(yy>0,risk,sample_weight=w),weighted_behavior_AP=float(np.mean([ap(yy==k,risk*(behavior==k),sample_weight=w) for k in [1,2,3]]))))
    (dest/'metrics.json').write_text(json.dumps(reports,indent=2));print('RESULT',scope,[r for r in reports if r['scope']==scope],flush=True)
