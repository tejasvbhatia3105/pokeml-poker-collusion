import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import time,json,numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import log_loss,mean_absolute_error
root=Path('artifacts/policy');paths=sorted((root/'actions').glob('*.parquet'))
while len(paths)<400:
 time.sleep(3);paths=sorted((root/'actions').glob('*.parquet'))
folds=json.loads(Path('artifacts/folds.json').read_text());tablefold={t:f['fold'] for f in folds for t in f['valid_tables']}
for i,p in enumerate(paths):tablefold.setdefault(p.stem,i%4)
(root/'table_folds.json').write_text(json.dumps(tablefold))
cols=json.loads((root/'feature_columns.json').read_text());parts=[]
for p in paths:
 d=pl.read_parquet(p).filter(pl.col('phase')=='development');rows=[]
 for street,n in enumerate([600,300,150,150]):
  z=d.filter(pl.col('street_no')==street)
  if len(z):rows.append(z.sample(n=min(n,len(z)),seed=414))
 parts.append(pl.concat(rows))
s=pl.concat(parts);X=s.select(cols).to_numpy();y=s['action_class'].to_numpy();size=s['log_bet_ratio'].to_numpy();g=np.array([tablefold[t] for t in s['table_id']]);reports=[]
for fold in range(4):
 va=g==fold;tr=~va;t=time.time()
 m=CatBoostClassifier(iterations=450,depth=6,learning_rate=.065,loss_function='MultiClass',l2_leaf_reg=10,thread_count=6,random_seed=310+fold,allow_writing_files=False,verbose=False)
 m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=60)
 m.save_model(str(root/f'action_fold{fold}.cbm'));pred=m.predict_proba(X[va]);loss=log_loss(y[va],pred,labels=[0,1,2,3])
 baseline=s.filter(pl.Series(va)).select([f'style_{k}' for k in range(4)]).to_numpy()
 report={'fold':fold,'action_trees':m.tree_count_,'action_logloss':loss,'style_only_logloss':log_loss(y[va],baseline,labels=[0,1,2,3]),'train_actions':int(tr.sum()),'validation_actions':int(va.sum())}
 print('action',report,'seconds',round(time.time()-t,1),flush=True)
 reg=CatBoostRegressor(iterations=350,depth=6,learning_rate=.07,loss_function='Huber:delta=0.5',l2_leaf_reg=10,thread_count=6,random_seed=510+fold,allow_writing_files=False,verbose=False)
 rt=tr&(y==3);rv=va&(y==3);reg.fit(X[rt],size[rt],eval_set=(X[rv],size[rv]),early_stopping_rounds=50);reg.save_model(str(root/f'size_fold{fold}.cbm'))
 report['size_mae']=mean_absolute_error(size[rv],reg.predict(X[rv]));report['size_trees']=reg.tree_count_;reports.append(report)
 (root/'ordinary_metrics.json').write_text(json.dumps(reports,indent=2));print('finished fold',fold,'seconds',round(time.time()-t,1),flush=True)
