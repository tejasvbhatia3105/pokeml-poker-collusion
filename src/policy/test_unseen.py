import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
root=Path('artifacts/policy');labs=pl.read_csv('data/development_labels.csv')
d=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='development').join(labs.lazy().select('pair_id','label','behavior_family'),on='pair_id').collect().sort('pair_id');cols=[c for c in d.columns if c not in ['pair_id','phase','player_1','player_2','table_id','label','behavior_family']]
X=d.select(cols).to_numpy();y=d['label'].to_numpy();b=d['behavior_family'].to_numpy();g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());nc=[c for c in cols if c.endswith(('_z','_w10_max','_w10_min'))];NX=d.select(nc).to_numpy();reports=[]
for family in ['directed_transfer','soft_play','coordinated_isolation']:
 pred=np.zeros(len(d));hybrid=np.zeros(len(d))
 for f in folds:
  va=np.isin(g,f['valid_tables']);tr=~va&(b!=family)
  m=CatBoostClassifier(iterations=600,depth=4,learning_rate=.045,loss_function='Logloss',l2_leaf_reg=10,verbose=False,thread_count=4,random_seed=318+f['fold'],allow_writing_files=False)
  m.fit(X[tr],y[tr]);pred[va]=m.predict_proba(X[va])[:,1]
  normals=tr&(y==0);med=np.median(NX[normals],axis=0);scale=np.maximum(np.quantile(NX[normals],.9,axis=0)-np.quantile(NX[normals],.1,axis=0),.01)
  novelty=np.sort(np.log1p(np.abs((NX-med)/scale)),axis=1)[:,-3:].mean(1)
  cal=LogisticRegression(C=1).fit(novelty[tr,None],y[tr]);npred=cal.predict_proba(novelty[va,None])[:,1]
  hybrid[va]=np.maximum(pred[va],npred)
 keep=(b==family)|(y==0)
 for name,p in [('binary',pred),('novelty_hybrid',hybrid)]:
  r={'heldout_family':family,'model':name,'AP':average_precision_score(y[keep],p[keep]),'AP_negative_weight50':average_precision_score(y[keep],p[keep],sample_weight=np.where(y[keep]>0,1,50))};reports.append(r);print(r,flush=True)
(root/'unseen_metrics.json').write_text(json.dumps(reports,indent=2))
