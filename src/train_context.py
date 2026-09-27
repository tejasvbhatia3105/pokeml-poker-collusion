import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import time,json,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from train_baseline import metric
while len(list(Path('artifacts/context_pairs').glob('*.parquet')))<400:time.sleep(3)
labs=pl.read_csv('data/development_labels.csv');ev=pl.read_csv('data/evaluation_pairs.csv');names=['none','directed_transfer','soft_play','coordinated_isolation']
s=pl.scan_parquet('artifacts/context_pairs/*.parquet')
d=s.filter(pl.col('phase')=='development').join(labs.lazy().select('pair_id','behavior_family'),on='pair_id').collect().sort('pair_id')
cols=[c for c in d.columns if c not in ['pair_id','phase','table_id','player_1','player_2','behavior_family']]
X=d.select(cols).to_numpy();y=np.array([names.index(n) for n in d['behavior_family']]);groups=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());oof=np.zeros((len(d),4));models=[]
for f in folds:
 va=np.isin(groups,f['valid_tables']);tr=~va
 m=CatBoostClassifier(iterations=1100,depth=5,learning_rate=.035,loss_function='MultiClass',l2_leaf_reg=9,verbose=False,thread_count=6,random_seed=820+f['fold'])
 m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=120)
 oof[va]=m.predict_proba(X[va]);models.append(m);m.save_model(f"artifacts/context_fold{f['fold']}.cbm")
 print(f['fold'],m.best_iteration_,metric(y[va],oof[va]),flush=True)
report=metric(y,oof);report['prevalence_weighted_pair_ap']={str(w):average_precision_score(y>0,1-oof[:,0],sample_weight=np.where(y>0,1,w)) for w in [1,10,50,100]}
print('OOF',report,flush=True)
Path('artifacts/context_metrics.json').write_text(json.dumps(report,indent=2));Path('artifacts/context_columns.json').write_text(json.dumps(cols))
pd.DataFrame(oof,columns=names).assign(pair_id=d['pair_id'].to_list(),truth=y).to_csv('artifacts/context_oof.csv',index=False)
pd.DataFrame({'feature':cols,'importance':np.mean([m.feature_importances_ for m in models],0)}).sort_values('importance',ascending=False).to_csv('artifacts/context_importance.csv',index=False)
out=[]
for i,path in enumerate(sorted(Path('artifacts/context_pairs').glob('*.parquet'))):
 z=pl.read_parquet(path).filter(pl.col('phase')=='evaluation').join(ev.select('pair_id'),on='pair_id')
 pr=np.mean([m.predict_proba(z.select(cols).to_numpy()) for m in models],axis=0)
 out.append(z.select('pair_id').with_columns([pl.Series(n,pr[:,i]) for i,n in enumerate(names)]))
pl.concat(out).sort('pair_id').write_csv('artifacts/context_eval.csv')
