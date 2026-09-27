import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,json
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
f=pl.read_parquet('artifacts/pair_features.parquet').filter(pl.col('phase')=='development').join(pl.read_csv('data/development_labels.csv'),on='pair_id').sort('pair_id')
cols=[c for c in f.columns if c.endswith(('_mean','_rate','_std','_top5'))]
X=f.select(cols).to_numpy();y=f['label'].to_numpy();b=f['behavior_family'].to_numpy();g=f['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());out=[]
for family in ['directed_transfer','soft_play','coordinated_isolation']:
 pred=np.zeros(len(f));nov=np.zeros(len(f))
 for fold in folds:
  va=np.isin(g,fold['valid_tables']);tr=~va&(b!=family)
  m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=8,verbose=False,thread_count=4,random_seed=191)
  m.fit(X[tr],y[tr]);pred[va]=m.predict_proba(X[va])[:,1]
  n=IsolationForest(n_estimators=200,max_samples=256,random_state=191,n_jobs=4)
  n.fit(X[tr&(y==0)]);nov[va]=-n.score_samples(X[va])
 select=(b==family)|(y==0)
 r={'omitted_family':family,'binary_unseen_AP':average_precision_score(y[select],pred[select]),'novelty_unseen_AP':average_precision_score(y[select],nov[select]),'binary_unseen_AP_negative_weight50':average_precision_score(y[select],pred[select],sample_weight=np.where(y[select]>0,1,50))}
 out.append(r);print(r,flush=True)
Path('artifacts/open_set_diagnostic.json').write_text(json.dumps(out,indent=2))
