import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,polars as pl,pandas as pd,time
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
names=['none','directed_transfer','soft_play','coordinated_isolation'];folds=json.loads(Path('artifacts/folds.json').read_text());allcols=json.loads(Path('artifacts/context_columns.json').read_text())
labs=pl.read_csv('data/development_labels.csv');posplayers=set(labs.filter(pl.col('label')==1)['player_1'].to_list()+labs.filter(pl.col('label')==1)['player_2'].to_list())
                                                                                                
top_by_fold=[]
for f in folds:
 m=CatBoostClassifier();m.load_model(f"artifacts/context_fold{f['fold']}.cbm");idx=np.argsort(-m.get_feature_importance())[:300];top_by_fold.append([allcols[i] for i in idx])
cols=[c for c in allcols if any(c in z for z in top_by_fold)]
d=pl.scan_parquet('artifacts/context_pairs/*.parquet').filter(pl.col('phase')=='development').select('pair_id','table_id','player_1','player_2',*cols).join(labs.lazy().select('pair_id','label'),on='pair_id',how='left').collect().sort('pair_id')
labeled=d.filter(pl.col('label').is_not_null());background=d.filter(pl.col('label').is_null()&~pl.col('player_1').is_in(list(posplayers))&~pl.col('player_2').is_in(list(posplayers)))
y=labeled['label'].to_numpy();g=labeled['table_id'].to_numpy();oof=np.zeros(len(y));models=[];t=time.time()
for f,fc in zip(folds,top_by_fold):
 va=np.isin(g,f['valid_tables']);tr=~va
 un=background.filter(~pl.col('table_id').is_in(f['valid_tables']));un=un.sample(n=min(14000,len(un)),seed=842+f['fold'])
 X=np.concatenate([labeled.filter(pl.Series(tr)).select(fc).to_numpy(),un.select(fc).to_numpy()]);Y=np.concatenate([y[tr],np.zeros(len(un))]);W=np.concatenate([np.ones(tr.sum()),np.full(len(un),.10)])
 m=CatBoostClassifier(iterations=700,depth=6,learning_rate=.035,loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=12,verbose=False,thread_count=6,random_seed=842+f['fold'],allow_writing_files=False)
 m.fit(X,Y,sample_weight=W,eval_set=(labeled.filter(pl.Series(va)).select(fc).to_numpy(),y[va]),early_stopping_rounds=100)
 oof[va]=m.predict_proba(labeled.filter(pl.Series(va)).select(fc).to_numpy())[:,1]
 m.save_model(f"artifacts/pu_fold{f['fold']}.cbm");models.append(m)
 print(f['fold'],'iter',m.best_iteration_,'AP',average_precision_score(y[va],oof[va]),'seconds',round(time.time()-t,1),flush=True)
report={'method':'PU surrogate-background training; each fold samples up to 14000 unlisted pairs with weight 0.10, confirmed pairs weight 1. Public positive players excluded from background. No evaluation labels used.','pair_ap':average_precision_score(y,oof),'weighted_pair_ap50':average_precision_score(y,oof,sample_weight=np.where(y>0,1,50))}
Path('artifacts/pu_metrics.json').write_text(json.dumps(report,indent=2));Path('artifacts/pu_columns.json').write_text(json.dumps(top_by_fold));print(report,flush=True)
pd.DataFrame({'pair_id':labeled['pair_id'].to_list(),'risk_score':oof,'truth':y}).to_csv('artifacts/pu_oof.csv',index=False)
ev=pl.read_csv('data/evaluation_pairs.csv').select('pair_id');parts=[]
for path in sorted(Path('artifacts/context_pairs').glob('*.parquet')):
 z=pl.read_parquet(path).filter(pl.col('phase')=='evaluation').join(ev,on='pair_id')
 p=np.mean([m.predict_proba(z.select(fc).to_numpy())[:,1] for m,fc in zip(models,top_by_fold)],axis=0)
 parts.append(z.select('pair_id').with_columns(pl.Series('risk_score',p)))
pl.concat(parts).sort('pair_id').write_csv('artifacts/pu_eval.csv')
