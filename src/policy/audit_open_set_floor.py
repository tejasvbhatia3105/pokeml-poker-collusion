import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,polars as pl,pandas as pd,time
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'open_set_floor';dest.mkdir(exist_ok=True);labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');d=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='development').join(labs.lazy(),on='pair_id').collect().sort('pair_id');cols=json.loads((root/'residual_columns.json').read_text());nc=[c for c in cols if c.endswith(('_z','_w10_max','_w10_min'))];X=d.select(cols).to_numpy();NX=d.select(nc).to_numpy();y=d['label'].to_numpy();b=d['behavior_family'].to_numpy();g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());reports=[];allpred=[];t=time.time()
for family in ['all_known','directed_transfer','soft_play','coordinated_isolation']:
 p=np.zeros(len(d));pn=np.zeros(len(d))
 for f in folds:
  va=np.isin(g,f['valid_tables']);tr=~va if family=='all_known' else ~va&(b!=family)
  if family=='all_known':
   m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));p[va]=1-m.predict_proba(X[va])[:,0]
  else:
   m=CatBoostClassifier(iterations=600,depth=4,learning_rate=.045,loss_function='Logloss',l2_leaf_reg=10,verbose=False,thread_count=4,random_seed=318+f['fold'],allow_writing_files=False);m.fit(X[tr],y[tr]);p[va]=m.predict_proba(X[va])[:,1]
  normal=tr&(y==0);med=np.median(NX[normal],axis=0);scale=np.maximum(np.quantile(NX[normal],.9,axis=0)-np.quantile(NX[normal],.1,axis=0),.01);score=np.sort(np.log1p(np.abs((NX-med)/scale)),axis=1)[:,-3:].mean(1);cal=LogisticRegression(C=1).fit(score[tr,None],y[tr]);pn[va]=cal.predict_proba(score[va,None])[:,1]
  if family=='all_known':np.savez(dest/f'calibrator_fold{f["fold"]}.npz',med=med,scale=scale,coef=cal.coef_,intercept=cal.intercept_,columns=np.array(nc))
 keep=np.ones(len(d),bool) if family=='all_known' else (y==0)|(b==family)
 for alpha in [0,.25,.5,.75,1]:
  risk=np.maximum(p,alpha*pn);reports.append({'withheld_family':family,'novelty_floor':alpha,'weighted_AP':ap(y[keep],risk[keep],sample_weight=np.where(y[keep]>0,1,50))})
 allpred.append(pd.DataFrame({'pair_id':d['pair_id'].to_list(),'withheld_family':family,'truth':y,'behavior':b,'base_risk':p,'novelty_probability':pn}));print(family,[r for r in reports if r['withheld_family']==family],'seconds',round(time.time()-t,1),flush=True)
pd.concat(allpred).to_csv(dest/'oof.csv',index=False);(dest/'metrics.json').write_text(json.dumps(reports,indent=2))
                                                                                                    
known={r['novelty_floor']:r['weighted_AP'] for r in reports if r['withheld_family']=='all_known'};choices=[]
for alpha in [0,.25,.5,.75,1]:
 v=[r['weighted_AP'] for r in reports if r['withheld_family']!='all_known' and r['novelty_floor']==alpha];choices.append({'floor':alpha,'known_loss':known[0]-known[alpha],'mean_withheld_AP':float(np.mean(v)),'worst_withheld_AP':float(min(v)),'eligible':known[0]-known[alpha]<=.005})
best=max([r for r in choices if r['eligible']],key=lambda r:r['mean_withheld_AP']);report={'choices':choices,'selected':best,'selection_caveat':'Same public families used to design and select this experiment; actual undisclosed-family performance remains unmeasured.'};(dest/'selection.json').write_text(json.dumps(report,indent=2));print('SELECTED',report,flush=True)
