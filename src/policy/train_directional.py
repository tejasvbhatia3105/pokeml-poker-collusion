\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap

root=Path('artifacts/policy'); dest=root/'directional'; C=pl.col
core=json.loads((root/'residual_columns.json').read_text())
labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family')
names=['none','directed_transfer','soft_play','coordinated_isolation']
folds=json.loads(Path('artifacts/folds.json').read_text())
parts=[]
for window in ['full','first_2000','last_2000']:
    folder=root/'pair_features' if window=='full' else root/'full_window_stress'/window/'pair_features'
    d=pl.scan_parquet(str(folder/'*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').collect()
    extra=pl.read_parquet(list((dest/window).glob('*.parquet'))).filter(C('phase')=='development')
    d=d.join(extra,on=['pair_id','phase'],how='left')
    assert d.select(pl.selectors.starts_with('dir_')).null_count().to_numpy().sum()==0
    d=d.join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible'))
    parts.append(d.with_columns(pl.lit(window).alias('window')))
d=pl.concat(parts).sort('window','pair_id')
new=[c for c in d.columns if c.startswith('dir_')]
y=np.array([names.index(b) for b in d['behavior_family']]); group=d['table_id'].to_numpy(); window=d['window'].to_numpy()
reports=[]; output=[]; t=time.time()
modes=['control','directional','combined']
for mode in modes:
    cols=core if mode=='control' else new if mode=='directional' else core+new
    (dest/f'{mode}_columns.json').write_text(json.dumps(cols))
    X=d.select(cols).to_numpy(); assert np.isfinite(X).all(); pred=np.zeros((len(d),4)); models=[]
    for f in folds:
        va=np.isin(group,f['valid_tables']); tr=~va&(window=='full')
        m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,
            thread_count=4,random_seed=991+f['fold'],verbose=False,allow_writing_files=False)
        m.fit(X[tr],y[tr]); pred[va]=m.predict_proba(X[va],thread_count=4)
        m.save_model(str(dest/f'{mode}_fold{f["fold"]}.cbm'));models.append(m)
        print('fit',mode,f['fold'],round(time.time()-t,1),flush=True)
    output.append(pd.DataFrame(pred,columns=names).assign(pair_id=d['pair_id'].to_list(),window=window,truth=y,model=mode))
    pd.DataFrame(dict(feature=cols,importance=np.mean([m.feature_importances_ for m in models],axis=0))).sort_values('importance',ascending=False).to_csv(dest/f'{mode}_importance.csv',index=False)
    for w in ['full','first_2000','last_2000']:
        keep=window==w; yy=y[keep]; risk=1-pred[keep,0]; weight=np.where(yy>0,1,50)
        reports.append(dict(model=mode,window=w,AP=ap(yy>0,risk),weighted_AP=ap(yy>0,risk,sample_weight=weight)))
    pd.concat(output).to_csv(dest/'oof.csv',index=False)
    (dest/'metrics.json').write_text(json.dumps(reports,indent=2));print('RESULT',reports[-3:],flush=True)
