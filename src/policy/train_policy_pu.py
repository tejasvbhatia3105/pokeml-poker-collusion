\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'policy_pu';dest.mkdir(exist_ok=True)
cols=json.loads((root/'residual_columns.json').read_text());labs=pl.read_csv('data/development_labels.csv')
allpairs=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='development').collect()
unknown=allpairs.join(labs.select('pair_id'),on='pair_id',how='anti').sort('pair_id').sample(n=20000,seed=5472)
unknown.select('pair_id','table_id').write_csv(dest/'unknown_background.csv')
UX=unknown.select(cols).to_numpy();ug=unknown['table_id'].to_numpy();parts=[]
for window in ['full','first_2000','last_2000']:
    d=pl.read_parquet(root/f'exposure_{window}.parquet').join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(pl.col('eligible')).with_columns(pl.lit(window).alias('window'));parts.append(d)
d=pl.concat(parts);names=['none','directed_transfer','soft_play','coordinated_isolation'];X=d.select(cols).to_numpy();y=np.array([names.index(b) for b in d['behavior_family']]);g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());orig=d['window'].to_numpy()=='full';base=np.zeros((len(d),4));t=time.time()
for f in folds:
    va=np.isin(g,f['valid_tables']);m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));base[va]=m.predict_proba(X[va])
reports=[]
for weight in [.03,.1]:
    pred=np.zeros_like(base)
    for f in folds:
        va=np.isin(g,f['valid_tables']);tr=~va&orig;ut=~np.isin(ug,f['valid_tables']);TX=np.concatenate([X[tr],UX[ut]]);ty=np.concatenate([y[tr],np.zeros(ut.sum(),dtype=int)]);w=np.concatenate([np.ones(tr.sum()),np.full(ut.sum(),weight)])
        m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=4,random_seed=991+f['fold'],verbose=False,allow_writing_files=False)
        m.fit(TX,ty,sample_weight=w);pred[va]=m.predict_proba(X[va]);m.save_model(str(dest/f'weight{weight}_fold{f["fold"]}.cbm'));print('fit',weight,f['fold'],round(time.time()-t,1),flush=True)
    pd.DataFrame(pred,columns=names).assign(pair_id=d['pair_id'].to_list(),window=d['window'].to_list(),truth=y).to_csv(dest/f'oof_weight{weight}.csv',index=False)
    for blend in [0,.5,1]:
        p=blend*pred+(1-blend)*base
        for window in ['full','first_2000','last_2000']:
            keep=d['window'].to_numpy()==window;yy=y[keep];risk=1-p[keep,0];behavior=p[keep,1:].argmax(1)+1;sw=np.where(yy>0,1,50)
            reports.append(dict(unknown_weight=weight,blend=blend,window=window,AP=ap(yy>0,risk),weighted_AP=ap(yy>0,risk,sample_weight=sw),weighted_behavior_AP=float(np.mean([ap(yy==k,risk*(behavior==k),sample_weight=sw) for k in [1,2,3]]))))
    (dest/'metrics.json').write_text(json.dumps(reports,indent=2));print('RESULT',weight,[r for r in reports if r['unknown_weight']==weight],flush=True)
