import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
from sklearn.covariance import LedoitWolf
root=Path('artifacts/policy');dest=root/'joint_model';dest.mkdir(exist_ok=True)
while len(list((root/'joint_pairs').glob('*.parquet')))<400:time.sleep(3)
labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');d=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='development').join(labs.lazy(),on='pair_id').collect().sort('pair_id');j=pl.scan_parquet(str(root/'joint_pairs/*.parquet')).filter(pl.col('phase')=='development').drop('phase','table_id').collect();d=d.join(j,on='pair_id').sort('pair_id');cols=[c for c in d.columns if c not in ['pair_id','phase','table_id','player_1','player_2','label','behavior_family']];names=['none','directed_transfer','soft_play','coordinated_isolation'];y=np.array([names.index(b) for b in d['behavior_family']]);g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());reports=[]
for mode in ['joint_only','core_plus_joint']:
 nc=[c for c in cols if c.startswith('joint_') and c.endswith(('_mean','_w10'))]
 if mode=='core_plus_joint':nc += [c for c in cols if c.endswith(('_z','_w10_max','_w10_min'))]
 X=d.select(nc).to_numpy();pred={n:np.zeros(len(d)) for n in ['max','top3','mahalanobis']}
 for f in folds:
  va=np.isin(g,f['valid_tables']);tr=~va&(y==0);med=np.median(X[tr],axis=0);scale=np.maximum(np.quantile(X[tr],.9,axis=0)-np.quantile(X[tr],.1,axis=0),.01);z=(X-med)/scale;z=np.sign(z)*np.log1p(np.abs(z));m=LedoitWolf().fit(z[tr]);pred['max'][va]=np.abs(z[va]).max(1);pred['top3'][va]=np.sort(np.abs(z[va]),axis=1)[:,-3:].mean(1);pred['mahalanobis'][va]=m.mahalanobis(z[va])
 for method,p in pred.items():
  for k in range(4):
   keep=np.ones(len(d),bool) if k==0 else (y==0)|(y==k);r={'features':mode,'method':method,'family':'all' if k==0 else names[k],'weighted_AP':ap(y[keep]>0,p[keep],sample_weight=np.where(y[keep]>0,1,50))};reports.append(r)
 print(mode,[r for r in reports if r['features']==mode],flush=True)
(dest/'novelty_metrics.json').write_text(json.dumps(reports,indent=2))
X=d.select(cols).to_numpy();oof=np.zeros((len(d),4));models=[];t=time.time()
for f in folds:
 va=np.isin(g,f['valid_tables']);tr=~va;m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=6,random_seed=991+f['fold'],verbose=False,allow_writing_files=False);m.fit(X[tr],y[tr]);oof[va]=m.predict_proba(X[va]);m.save_model(str(dest/f'fold{f["fold"]}.cbm'));models.append(m);print('fit',f['fold'],round(time.time()-t,1),flush=True)
pd.DataFrame(oof,columns=names).assign(pair_id=d['pair_id'].to_list(),truth=y).to_csv(dest/'oof.csv',index=False);(dest/'columns.json').write_text(json.dumps(cols));old=pd.read_csv(root/'residual_oof.csv').sort_values('pair_id');assert old.pair_id.to_list()==d['pair_id'].to_list();r=[]
for alpha in [0,.25,.5,.75,1]:
 p=alpha*oof+(1-alpha)*old[names].values;risk=1-p[:,0];pred=p[:,1:].argmax(1)+1;w=np.where(y>0,1,50);r.append({'joint_weight':alpha,'pair_AP':ap(y>0,risk),'weighted_AP':ap(y>0,risk,sample_weight=w),'weighted_behavior':np.mean([ap(y==k,risk*(pred==k),sample_weight=w) for k in [1,2,3]])})
(dest/'metrics.json').write_text(json.dumps(r,indent=2));print('MODEL',r,flush=True)
pd.DataFrame({'feature':cols,'importance':np.mean([m.feature_importances_ for m in models],0)}).sort_values('importance',ascending=False).to_csv(dest/'importance.csv',index=False)
