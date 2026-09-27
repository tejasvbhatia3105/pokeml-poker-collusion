\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
C=pl.col;root=Path('artifacts/policy');dest=root/'event_detector';dest.mkdir(exist_ok=True)
labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');ev=pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('event'))
h=pl.scan_parquet(str(root/'hand_features/*.parquet')).filter(C('phase')=='development').join(labs.lazy().select('pair_id'),on='pair_id').collect().sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')];h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
d=pl.read_parquet('artifacts/dev_detail.parquet').join(labs,on='pair_id').join(ev,on=['pair_id','hand_id'],how='left').with_columns(C('event').fill_null(0)).join(h.select('pair_id','hand_id','time_index',*add),on=['pair_id','hand_id']).sort('pair_id','hand_id')
cols=[c for c in json.loads(Path('artifacts/rank_columns.json').read_text()) if c!='relative_time']+add
eligible=(C('label')==0)|(C('event')==1);d=d.with_columns(eligible.alias('training_eligible'));d=d.with_columns((C('training_eligible').sum().over('pair_id')).alias('n_training_hands'));X=d.select(cols).to_numpy();y=d['event'].to_numpy();g=d['table_id'].to_numpy();weights=1/d['n_training_hands'].to_numpy();allowed=d['training_eligible'].to_numpy();oof=np.zeros(len(d));folds=json.loads(Path('artifacts/folds.json').read_text());t=time.time()
for f in folds:
 va=np.isin(g,f['valid_tables']);tr=~va&allowed;m=CatBoostClassifier(iterations=600,depth=5,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=10,thread_count=5,random_seed=1512+f['fold'],verbose=False,allow_writing_files=False);m.fit(X[tr],y[tr],sample_weight=weights[tr]);oof[va]=m.predict_proba(X[va])[:,1];m.save_model(str(dest/f'fold{f["fold"]}.cbm'));print('fold',f['fold'],'seconds',round(time.time()-t,1),flush=True)
(dest/'columns.json').write_text(json.dumps(cols));d=d.select('pair_id','hand_id','label','behavior_family','time_index','event').with_columns(pl.Series('event_score',oof));d.write_parquet(dest/'oof_hand_scores.parquet');reports=[];old=pl.read_csv(root/'residual_oof.csv').select('pair_id',(1-C('none')).alias('v4_risk'))
for window,mask in [('full',pl.lit(True)),('first_2000',C('time_index')<2000),('last_2000',C('time_index')>=1000)]:
 z=d.filter(mask);r=z.group_by('pair_id').agg(C('label').first(),C('event').sum().alias('evidence_left'),*[C('event_score').top_k(k).mean().alias(f'top{k}') for k in [1,3,5,10]]).filter((C('label')==0)|(C('evidence_left')>0)).join(old,on='pair_id');yy=r['label'].to_numpy();w=np.where(yy>0,1,50)
 for k in [1,3,5,10]:
  for alpha in [0,.25,.5,.75,1]:
   pred=alpha*r[f'top{k}'].to_numpy()+(1-alpha)*r['v4_risk'].to_numpy()
                                                                                                           
   if window!='full' and alpha!=1:continue
   reports.append({'window':window,'top_k':k,'event_weight':alpha,'pair_AP':ap(yy,pred),'weighted_AP':ap(yy,pred,sample_weight=w)})
 print(window,[x for x in reports if x['window']==window and x['event_weight']==1],flush=True)
(dest/'metrics.json').write_text(json.dumps(reports,indent=2));print('BEST_FULL',max([r for r in reports if r['window']=='full'],key=lambda x:x['weighted_AP']),flush=True)
