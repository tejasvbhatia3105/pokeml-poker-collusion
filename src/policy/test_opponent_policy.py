import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from opponent_inputs import enrich,COLUMNS
root=Path('artifacts/policy');dest=root/'opponent_policy';dest.mkdir(exist_ok=True);basecols=json.loads((root/'feature_columns.json').read_text());cols=basecols+COLUMNS;folds=json.loads((root/'table_folds.json').read_text());parts=[];t=time.time()
for path in sorted((root/'actions').glob('*.parquet')):
 d=enrich(pl.read_parquet(path).filter(pl.col('phase')=='development'));rows=[]
 for street,n in enumerate([600,300,150,150]):
  z=d.filter(pl.col('street_no')==street)
  if len(z):rows.append(z.sample(n=min(n,len(z)),seed=414))
 parts.append(pl.concat(rows))
s=pl.concat(parts);X=s.select(cols).to_numpy();oldX=s.select(basecols).to_numpy();y=s['action_class'].to_numpy();g=np.array([folds[t] for t in s['table_id']]);reports=[];print('prepared',len(s),'seconds',round(time.time()-t,1),flush=True)
for f in range(4):
 va=g==f;tr=~va;m=CatBoostClassifier(iterations=450,depth=6,learning_rate=.065,loss_function='MultiClass',l2_leaf_reg=10,thread_count=6,random_seed=310+f,allow_writing_files=False,verbose=False);m.fit(X[tr],y[tr]);m.save_model(str(dest/f'action_fold{f}.cbm'));new=m.predict_proba(X[va]);old=CatBoostClassifier();old.load_model(str(root/f'action_fold{f}.cbm'));baseline=old.predict_proba(oldX[va]);r={'fold':f,'old_logloss':log_loss(y[va],baseline),'opponent_logloss':log_loss(y[va],new)};reports.append(r);print(r,'seconds',round(time.time()-t,1),flush=True)
(dest/'metrics.json').write_text(json.dumps(reports,indent=2));(dest/'columns.json').write_text(json.dumps(cols))
