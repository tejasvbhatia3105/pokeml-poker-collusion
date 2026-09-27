import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap

root=Path('artifacts/policy');dest=root/'directional';C=pl.col
core=json.loads((root/'residual_columns.json').read_text())
labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');parts=[]
for window in ['full','first_2000','last_2000']:
    folder=root/'pair_features' if window=='full' else root/'full_window_stress'/window/'pair_features'
    d=pl.scan_parquet(str(folder/'*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').collect()
    extra=pl.read_parquet(list((dest/window).glob('*.parquet'))).filter(C('phase')=='development')
    d=d.join(extra,on=['pair_id','phase']).join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible'))
    parts.append(d.with_columns(pl.lit(window).alias('window')))
d=pl.concat(parts).sort('window','pair_id');cols=core+[c for c in d.columns if c.startswith('dir_')]
X=d.select(cols).to_numpy();y=d['label'].to_numpy();b=d['behavior_family'].to_numpy();g=d['table_id'].to_numpy();w=d['window'].to_numpy()
folds=json.loads(Path('artifacts/folds.json').read_text());rows=[];output=[];t=time.time()
old=pd.read_csv(root/'crossed_validation/oof.csv');old=old[old.training=='full_training']
for family in ['directed_transfer','soft_play','coordinated_isolation']:
    pred=np.zeros(len(d))
    for f in folds:
        va=np.isin(g,f['valid_tables']);tr=~va&(w=='full')&(b!=family)
        m=CatBoostClassifier(iterations=600,depth=4,learning_rate=.045,loss_function='Logloss',l2_leaf_reg=10,
            verbose=False,thread_count=4,random_seed=318+f['fold'],allow_writing_files=False)
        m.fit(X[tr],y[tr]);pred[va]=m.predict_proba(X[va],thread_count=4)[:,1]
        m.save_model(str(dest/f'unseen_{family}_fold{f["fold"]}.cbm'));print('unseen',family,f['fold'],round(time.time()-t,1),flush=True)
    result=pd.DataFrame(dict(pair_id=d['pair_id'].to_list(),window=w,truth=y,behavior=b,score=pred,withheld_family=family))
    result=result.merge(old[old.withheld_family==family][['pair_id','window','base']],on=['pair_id','window'],validate='one_to_one')
    output.append(result)
    for window,q in result.groupby('window'):
        q=q[(q.truth==0)|(q.behavior==family)]
        for mode,score in [('control',q.base),('directional_combined',q.score)]:
            rows.append(dict(family=family,window=window,model=mode,weighted_AP=ap(q.truth,score,sample_weight=np.where(q.truth>0,1,50))))
    pd.concat(output).to_csv(dest/'unseen_oof.csv',index=False);pd.DataFrame(rows).to_csv(dest/'unseen_metrics.csv',index=False)
    print('UNSEEN RESULT',rows[-6:],flush=True)
