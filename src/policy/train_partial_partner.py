import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap

root=Path('artifacts/policy');dest=root/'partial_partner';C=pl.col
core=json.loads((root/'residual_columns.json').read_text());labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');parts=[]
for window in ['full','first_2000','last_2000']:
    folder=root/'pair_features' if window=='full' else root/'full_window_stress'/window/'pair_features'
    d=pl.scan_parquet(str(folder/'*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').collect()
    extra=pl.read_parquet(list((dest/window).glob('*.parquet'))).filter(C('phase')=='development')
    d=d.join(extra,on=['pair_id','phase'],how='left');new=[c for c in extra.columns if c.startswith('partial_')]
    assert d.select(new).null_count().to_numpy().sum()==0
    d=d.join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible'));parts.append(d.with_columns(pl.lit(window).alias('window')))
d=pl.concat(parts).sort('window','pair_id');cols=core+new;(dest/'columns.json').write_text(json.dumps(cols))
names=['none','directed_transfer','soft_play','coordinated_isolation'];X=d.select(cols).to_numpy();y=d['label'].to_numpy();b=d['behavior_family'].to_numpy();ym=np.array([names.index(s) for s in b]);g=d['table_id'].to_numpy();w=d['window'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text())
baseline=pd.read_csv(root/'crossed_validation/oof.csv');baseline=baseline[baseline.training=='full_training'];reports=[];outputs=[];t=time.time()
for family in ['all_known',*names[1:]]:
    pred=np.zeros(len(d));control=np.zeros(len(d));probs=np.zeros((len(d),4));models=[]
    for f in folds:
        va=np.isin(g,f['valid_tables']);tr=~va&(w=='full')&((b!=family) if family!='all_known' else True)
        params=dict(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',random_seed=991+f['fold']) if family=='all_known' else dict(iterations=600,depth=4,learning_rate=.045,loss_function='Logloss',random_seed=318+f['fold'])
        m=CatBoostClassifier(**params,l2_leaf_reg=10,thread_count=4,verbose=False,allow_writing_files=False);m.fit(X[tr],ym[tr] if family=='all_known' else y[tr]);m.save_model(str(dest/f'{family}_fold{f["fold"]}.cbm'));models.append(m)
        p=m.predict_proba(X[va],thread_count=4)
        if family=='all_known':
            probs[va]=p;pred[va]=1-p[:,0];old=CatBoostClassifier();old.load_model(str(root/f'directional/control_fold{f["fold"]}.cbm'));control[va]=1-old.predict_proba(d.filter(pl.Series(va)).select(core).to_numpy(),thread_count=4)[:,0]
        else:pred[va]=p[:,1]
        print('partial fit',family,f['fold'],round(time.time()-t,1),flush=True)
    out=pd.DataFrame(dict(pair_id=d['pair_id'].to_list(),window=w,truth=y,behavior=b,withheld_family=family,score=pred))
    if family=='all_known':
        out['baseline']=control
        for k,name in enumerate(names):out[name]=probs[:,k]
    else:out=out.merge(baseline[baseline.withheld_family==family][['pair_id','window','base']].rename(columns={'base':'baseline'}),on=['pair_id','window'],validate='one_to_one')
    outputs.append(out)
    for window,q in out.groupby('window'):
        if family!='all_known':q=q[(q.truth==0)|(q.behavior==family)]
        for name,score in [('control',q.baseline),('partial',q.score)]:reports.append(dict(family=family,window=window,model=name,AP=ap(q.truth,score),weighted_AP=ap(q.truth,score,sample_weight=np.where(q.truth>0,1,50))))
    pd.concat(outputs).to_csv(dest/'oof.csv',index=False);pd.DataFrame(reports).to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(dict(feature=cols,importance=np.mean([m.feature_importances_ for m in models],axis=0))).sort_values('importance',ascending=False).to_csv(dest/f'{family}_importance.csv',index=False)
    print('RESULT',reports[-6:],flush=True)
