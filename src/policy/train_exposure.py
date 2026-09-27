import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'exposure_model';dest.mkdir(exist_ok=True);names=['none','directed_transfer','soft_play','coordinated_isolation'];cols=json.loads((root/'residual_columns.json').read_text());folds=json.loads(Path('artifacts/folds.json').read_text());parts=[]
for name in ['full','first_2000','last_2000']:
 d=pl.read_parquet(root/f'exposure_{name}.parquet').join(pl.read_parquet(root/f'exposure_{name}_eligibility.parquet'),on='pair_id').filter(pl.col('eligible')).with_columns(pl.lit(name).alias('window'));parts.append(d)
d=pl.concat(parts).with_columns((1/pl.len().over('pair_id')).alias('sample_weight'));X=d.select(cols).to_numpy();y=np.array([names.index(b) for b in d['behavior_family']]);weights=d['sample_weight'].to_numpy();g=d['table_id'].to_numpy();oof=np.zeros((len(d),4));t=time.time()
for f in folds:
 va=np.isin(g,f['valid_tables']);tr=~va
 m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=6,random_seed=991+f['fold'],verbose=False,allow_writing_files=False)
 m.fit(X[tr],y[tr],sample_weight=weights[tr]);oof[va]=m.predict_proba(X[va]);m.save_model(str(dest/f'fold{f["fold"]}.cbm'));print('fold',f['fold'],'seconds',round(time.time()-t,1),flush=True)
pd.DataFrame(oof,columns=names).assign(pair_id=d['pair_id'].to_list(),window=d['window'].to_list(),truth=y).to_csv(dest/'oof.csv',index=False)
                                                                                  
original=np.zeros_like(oof)
for f in folds:
 va=np.isin(g,f['valid_tables']);m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));original[va]=m.predict_proba(X[va])
reports=[]
for alpha in [0,.25,.5,.75,1]:
 p=alpha*oof+(1-alpha)*original
 for window in ['full','first_2000','last_2000']:
  keep=(d['window']==window).to_numpy();yy=y[keep];risk=1-p[keep,0];pred=p[keep,1:].argmax(1)+1;w=np.where(yy>0,1,50)
  report={'exposure_weight':alpha,'window':window,'pair_ap':ap(yy>0,risk),'weighted_pair_ap':ap(yy>0,risk,sample_weight=w),'weighted_behavior_map':np.mean([ap(yy==k,risk*(pred==k),sample_weight=w) for k in [1,2,3]])};reports.append(report)
  print(report,flush=True)
(dest/'metrics.json').write_text(json.dumps(reports,indent=2))
