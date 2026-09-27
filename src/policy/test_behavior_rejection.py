\
\
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
import importlib.util
spec=importlib.util.spec_from_file_location('reference_metric','artifacts/reference_metric/reference_metric.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

root=Path('artifacts/policy');dest=root/'behavior_rejection';dest.mkdir(exist_ok=True);C=pl.col
core=json.loads((root/'residual_columns.json').read_text());nc=[c for c in core if c.endswith('_z')]
labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');parts=[]
names=['none','directed_transfer','soft_play','coordinated_isolation']
for window in ['full','first_2000','last_2000']:
    folder=root/'pair_features' if window=='full' else root/'full_window_stress'/window/'pair_features'
    d=pl.scan_parquet(str(folder/'*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').collect()
    d=d.join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible'));parts.append(d.with_columns(pl.lit(window).alias('window')))
d=pl.concat(parts).sort('window','pair_id');X=d.select(core).to_numpy();NX=d.select(nc).to_numpy()
y=np.array([names.index(b) for b in d['behavior_family']]);g=d['table_id'].to_numpy();w=d['window'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text())
quantiles=[-1,0,.01,.05,.10,.20];records=[];outputs=[];t=time.time()
for family in ['all_known',*names[1:]]:
    excluded=names.index(family) if family!='all_known' else -1
    risk=np.zeros(len(d));assigned=np.zeros(len(d),int);similarity=np.zeros(len(d));thresholds=np.zeros((len(d),len(quantiles)))
    for f in folds:
        va=np.isin(g,f['valid_tables']);tr=~va&(w=='full')&(y!=excluded)
        m=CatBoostClassifier()
        if family=='all_known':m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'))
        else:
            m=CatBoostClassifier(iterations=600,depth=4,learning_rate=.045,loss_function='MultiClass',l2_leaf_reg=10,thread_count=4,random_seed=318+f['fold'],verbose=False,allow_writing_files=False)
            path=dest/f'{family}_fold{f["fold"]}.cbm'
            if path.exists():m.load_model(str(path))
            else:m.fit(X[tr],y[tr]);m.save_model(str(path))
        probs=m.predict_proba(X[va],thread_count=4);classes=m.classes_.astype(int);normal_index=np.where(classes==0)[0][0]
        risk[va]=1-probs[:,normal_index];targets=classes!=0;assigned[va]=classes[targets][np.argmax(probs[:,targets],axis=1)]
        normal=NX[tr&(y==0)];med=np.median(normal,axis=0);scale=np.maximum(np.quantile(normal,.9,axis=0)-np.quantile(normal,.1,axis=0),.01)
        z=(NX-med)/scale;z=np.sign(z)*np.log1p(abs(z));z/=np.linalg.norm(z,axis=1,keepdims=True).clip(1e-8)
        protos=[];qs=[];present=[]
        for k in classes[targets]:
            known=z[tr&(y==k)];total=known.sum(0);proto=total/max(np.linalg.norm(total),1e-8)
            loo=total[None,:]-known;loo/=np.linalg.norm(loo,axis=1,keepdims=True).clip(1e-8);calibration=np.sum(known*loo,axis=1)
            cuts=np.quantile(calibration,np.maximum(quantiles,0));cuts[0]=-2
            chosen=va&(assigned==k);similarity[chosen]=z[chosen]@proto;thresholds[chosen]=cuts
            protos.append(proto);qs.append(cuts);present.append(k)
        np.savez(dest/f'{family}_fold{f["fold"]}_prototype.npz',columns=np.array(nc),median=med,scale=scale,classes=np.array(present),prototypes=np.array(protos),thresholds=np.array(qs),quantiles=np.array(quantiles))
        print('behavior',family,f['fold'],round(time.time()-t,1),flush=True)
    for i,q in enumerate(quantiles):
        pred=np.where((risk>=.01)&(similarity>=thresholds[:,i]),assigned,0)
        for window in ['full','first_2000','last_2000']:
            keep=w==window; knownclasses=[k for k in [1,2,3] if k!=excluded];yy=y[keep];pp=pred[keep];rr=risk[keep];weights=np.where(yy==0,50,1)
            known=np.isin(yy,knownclasses);novel=yy==excluded
            records.append(dict(withheld_family=family,window=window,quantile=q,
                behavior_MAP=float(np.mean([reference._average_precision(yy==k,rr*(pp==k)) for k in knownclasses])),
                weighted_behavior_MAP=float(np.mean([ap(yy==k,rr*(pp==k),sample_weight=weights) for k in knownclasses])),
                known_rejected=int((known&(rr>=.01)&(pp==0)).sum()),unknown_rejected=int((novel&(rr>=.01)&(pp==0)).sum()),
                known_alerts=int((known&(rr>=.01)).sum()),unknown_alerts=int((novel&(rr>=.01)).sum())))
    outputs.append(pd.DataFrame(dict(pair_id=d['pair_id'].to_list(),window=w,truth=y,risk=risk,assigned=assigned,similarity=similarity,withheld_family=family,**{f'threshold_{q}':thresholds[:,i] for i,q in enumerate(quantiles)})))
    pd.concat(outputs).to_csv(dest/'oof.csv',index=False);pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    print('BEHAVIOR RESULT',pd.DataFrame(records).query('withheld_family == @family and quantile in [-1, 0, .01]').to_string(index=False),flush=True)
