\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
from pathlib import Path
import json, time
import numpy as np
import polars as pl
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap

root = Path('artifacts/policy'); dest = root/'relationship_context'
dest.mkdir(exist_ok=True); (dest/'features').mkdir(exist_ok=True)
corecols = json.loads((root/'residual_columns.json').read_text())
cc = [c for c in corecols if c.endswith(('_z','_w10_max','_w10_min'))]
C=pl.col; t=time.time()
for i,path in enumerate(sorted((root/'pair_features').glob('*.parquet'))):
    out=dest/'features'/path.name
    if out.exists(): continue
    d=pl.read_parquet(path)
    endpoints=pl.concat([d.select('pair_id','phase',C(role).alias('player'),*cc) for role in ['player_1','player_2']])
    group=['phase','player']
    n=pl.len().over(group)
    endpoints=endpoints.with_columns(*[(C(c)-(C(c).sum().over(group)-C(c))/(n-1)).alias(c+'_excess') for c in cc],
        *[((C(c).rank(method='average').over(group)-1)/(n-1)).alias(c+'_pct') for c in cc])
    stats=[]
    for c in cc:
        stats += [C(c+'_excess').min().alias('peer_'+c+'_excess_min'),
            C(c+'_excess').max().alias('peer_'+c+'_excess_max'),
            C(c+'_pct').min().alias('peer_'+c+'_pct_min'),
            C(c+'_pct').max().alias('peer_'+c+'_pct_max')]
    f=endpoints.group_by('pair_id','phase').agg(stats).with_columns(pl.selectors.float().cast(pl.Float32))
    f.write_parquet(out,compression='zstd')
    if i%100==0: print('features',i,round(time.time()-t,1),flush=True)

labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family')
d=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').join(pl.scan_parquet(str(dest/'features/*.parquet')),on=['pair_id','phase']).collect().sort('pair_id')
newcols=[c for c in d.columns if c.startswith('peer_')]
cols=corecols+newcols
X=d.select(cols).to_numpy(); assert np.isfinite(X).all()
names=['none','directed_transfer','soft_play','coordinated_isolation']
y=np.array([names.index(b) for b in d['behavior_family']]);g=d['table_id'].to_numpy()
folds=json.loads(Path('artifacts/folds.json').read_text())
pred=np.zeros((len(d),4)); models=[]
for f in folds:
    va=np.isin(g,f['valid_tables']);tr=~va
    m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',
        l2_leaf_reg=10,thread_count=4,random_seed=991+f['fold'],verbose=False,allow_writing_files=False)
    m.fit(X[tr],y[tr]); pred[va]=m.predict_proba(X[va]);models.append(m)
    m.save_model(str(dest/f'fold{f["fold"]}.cbm'))
    print('fit',f['fold'],round(time.time()-t,1),flush=True)
pd.DataFrame(pred,columns=names).assign(pair_id=d['pair_id'].to_list(),truth=y).to_csv(dest/'oof.csv',index=False)
(dest/'columns.json').write_text(json.dumps(cols))
old=pd.read_csv(root/'residual_oof.csv').set_index('pair_id').loc[d['pair_id'].to_list(),names].to_numpy()
reports=[]
for alpha in [0,.25,.5,.75,1]:
    p=alpha*pred+(1-alpha)*old;risk=1-p[:,0];behavior=p[:,1:].argmax(1)+1;w=np.where(y>0,1,50)
    reports.append(dict(context_weight=alpha,AP=ap(y>0,risk),weighted_AP=ap(y>0,risk,sample_weight=w),
        weighted_behavior_AP=float(np.mean([ap(y==k,risk*(behavior==k),sample_weight=w) for k in [1,2,3]]))))
(dest/'metrics.json').write_text(json.dumps(reports,indent=2))
pd.DataFrame(dict(feature=cols,importance=np.mean([m.feature_importances_ for m in models],axis=0))).sort_values('importance',ascending=False).to_csv(dest/'importance.csv',index=False)
print('RESULT',reports,flush=True)
