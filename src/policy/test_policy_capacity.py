import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','8')
from pathlib import Path
import time,json,numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
root=Path('artifacts/policy'); cols=json.loads((root/'feature_columns.json').read_text()); tf=json.loads((root/'table_folds.json').read_text())
per=[int(x) for x in sys.argv[1].split(',')] if len(sys.argv)>1 else [600,300,150,150]                                
iters=int(sys.argv[2]) if len(sys.argv)>2 else 450; depth=int(sys.argv[3]) if len(sys.argv)>3 else 6
parts=[]
for p in sorted((root/'actions').glob('*.parquet')):
    d=pl.read_parquet(p).filter(pl.col('phase')=='development'); rows=[]
    for street,n in enumerate(per):
        z=d.filter(pl.col('street_no')==street)
        if len(z): rows.append(z.sample(n=min(n,len(z)),seed=414))
    parts.append(pl.concat(rows))
s=pl.concat(parts); X=s.select(cols).to_numpy(); y=s['action_class'].to_numpy(); g=np.array([tf[t] for t in s['table_id']])
va=g==0; tr=~va; t=time.time()
m=CatBoostClassifier(iterations=iters,depth=depth,learning_rate=.065,loss_function='MultiClass',l2_leaf_reg=10,thread_count=8,random_seed=310,allow_writing_files=False,verbose=False)
m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=60)
print(json.dumps({'per_street':per,'iters':iters,'depth':depth,'train':int(tr.sum()),'valid':int(va.sum()),'trees':m.tree_count_,'logloss':log_loss(y[va],m.predict_proba(X[va]),labels=[0,1,2,3]),'seconds':round(time.time()-t)}),flush=True)
m.save_model(f'artifacts/policy_big/probe_{"_".join(map(str,per))}_{iters}_{depth}.cbm')
