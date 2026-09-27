import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'binary_exposure';dest.mkdir(exist_ok=True);cols=json.loads((root/'residual_columns.json').read_text());folds=json.loads(Path('artifacts/folds.json').read_text());parts=[]
for name in ['full','first_2000','last_2000']:
 d=pl.read_parquet(root/f'exposure_{name}.parquet').join(pl.read_parquet(root/f'exposure_{name}_eligibility.parquet'),on='pair_id').filter(pl.col('eligible')).with_columns(pl.lit(name).alias('window'));parts.append(d)
d=pl.concat(parts).with_columns((1/pl.len().over('pair_id')).alias('sample_weight'));X=d.select(cols).to_numpy();y=d['label'].to_numpy();weights=d['sample_weight'].to_numpy();g=d['table_id'].to_numpy();oof=np.zeros(len(d));old=np.zeros(len(d));t=time.time()
for f in folds:
 va=np.isin(g,f['valid_tables']);tr=~va;m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=10,thread_count=4,random_seed=991+f['fold'],verbose=False,allow_writing_files=False);m.fit(X[tr],y[tr],sample_weight=weights[tr]);oof[va]=m.predict_proba(X[va])[:,1];m.save_model(str(dest/f'fold{f["fold"]}.cbm'));m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));old[va]=1-m.predict_proba(X[va])[:,0];print('fold',f['fold'],round(time.time()-t,1),flush=True)
pd.DataFrame({'pair_id':d['pair_id'].to_list(),'window':d['window'].to_list(),'truth':y,'binary_risk':oof,'v4_risk':old}).to_csv(dest/'oof.csv',index=False);r=[]
for alpha in [0,.25,.5,.75,1]:
 for window in ['full','first_2000','last_2000']:
  keep=(d['window']==window).to_numpy();risk=(alpha*oof+(1-alpha)*old)[keep];yy=y[keep];r.append({'binary_weight':alpha,'window':window,'AP':ap(yy,risk),'weighted_AP':ap(yy,risk,sample_weight=np.where(yy>0,1,50))})
print(r,flush=True);(dest/'metrics.json').write_text(json.dumps(r,indent=2))
