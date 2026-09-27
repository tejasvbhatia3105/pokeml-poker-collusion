import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4')
sys.path.insert(0,'src')
from pathlib import Path
import time,json,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.covariance import LedoitWolf
from train_baseline import metric
root=Path('artifacts/policy')
while len(list((root/'pair_features').glob('*.parquet')))<400:time.sleep(3)
names=['none','directed_transfer','soft_play','coordinated_isolation'];labs=pl.read_csv('data/development_labels.csv')
d=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='development').join(labs.lazy().select('pair_id','behavior_family'),on='pair_id').collect().sort('pair_id')
cols=[c for c in d.columns if c not in ['pair_id','phase','player_1','player_2','table_id','behavior_family']]
y=np.array([names.index(n) for n in d['behavior_family']]);groups=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text())
                                                                                         
nc=[c for c in cols if c.endswith(('_z','_w10_max','_w10_min'))];X=d.select(nc).to_numpy();pred={n:np.zeros(len(d)) for n in ['mahalanobis','max','top3']}
for f in folds:
 va=np.isin(groups,f['valid_tables']);tr=~va&(y==0)
 med=np.median(X[tr],axis=0);scale=np.maximum(np.quantile(X[tr],.9,axis=0)-np.quantile(X[tr],.1,axis=0),.01)
 z=(X-med)/scale;z=np.sign(z)*np.log1p(np.abs(z));m=LedoitWolf().fit(z[tr]);pred['mahalanobis'][va]=m.mahalanobis(z[va]);pred['max'][va]=np.abs(z[va]).max(1);pred['top3'][va]=np.sort(np.abs(z[va]),axis=1)[:,-3:].mean(1)
r=[]
for n,p in pred.items():
 for k in range(4):
  keep=np.ones(len(y),bool) if k==0 else (y==0)|(y==k)
  r.append({'method':n,'family':'all' if k==0 else names[k],'AP':average_precision_score(y[keep]>0,p[keep]),'AP_negative_weight50':average_precision_score(y[keep]>0,p[keep],sample_weight=np.where(y[keep]>0,1,50))})
(root/'novelty_metrics.json').write_text(json.dumps(r,indent=2));print('NOVELTY',r,flush=True)
                                                                        
                                                                         
context=pl.scan_parquet('artifacts/context_pairs/*.parquet').filter(pl.col('phase')=='development').join(labs.lazy().select('pair_id'),on='pair_id').collect().sort('pair_id')
assert d['pair_id'].to_list()==context['pair_id'].to_list()
for mode in ['residual','combined']:
 features=d.select(cols)
 if mode=='combined':
  cx=[c for c in context.columns if c not in ['pair_id','phase','player_1','player_2','table_id']]
  features=features.hstack(context.select(cx))
 X=features.to_numpy();oof=np.zeros((len(d),4));models=[];t=time.time()
 for f in folds:
  va=np.isin(groups,f['valid_tables']);tr=~va
  m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=6,random_seed=991+f['fold'],verbose=False,allow_writing_files=False)
  m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=100)
  oof[va]=m.predict_proba(X[va]);m.save_model(str(root/f'{mode}_fold{f["fold"]}.cbm'));models.append(m)
  print(mode,f['fold'],m.tree_count_,metric(y[va],oof[va]),'seconds',round(time.time()-t,1),flush=True)
 report=metric(y,oof);report['weighted_pair_ap']={str(w):average_precision_score(y>0,1-oof[:,0],sample_weight=np.where(y>0,1,w)) for w in [1,10,50,100]}
 (root/f'{mode}_metrics.json').write_text(json.dumps(report,indent=2));(root/f'{mode}_columns.json').write_text(json.dumps(features.columns));print(mode,'OOF',report,flush=True)
 pd.DataFrame(oof,columns=names).assign(pair_id=d['pair_id'].to_list(),truth=y).to_csv(root/f'{mode}_oof.csv',index=False)
 pd.DataFrame({'feature':features.columns,'importance':np.mean([m.feature_importances_ for m in models],axis=0)}).sort_values('importance',ascending=False).to_csv(root/f'{mode}_importance.csv',index=False)
