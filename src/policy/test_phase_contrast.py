\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl,pandas as pd
from sklearn.metrics import average_precision_score as ap
from catboost import CatBoostClassifier
C=pl.col;root=Path('artifacts/policy');dest=root/'phase_contrast';dest.mkdir(exist_ok=True);labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');cols=json.loads((root/'residual_columns.json').read_text());nc=[c for c in cols if c.endswith(('_z','_w10_max','_w10_min'))];parts=[];allparts=[]
for path in sorted((root/'pair_features').glob('*.parquet')):
 f=pl.read_parquet(path);ref=f.select('pair_id',pl.when(C('phase')=='development').then(pl.lit('evaluation')).otherwise(pl.lit('development')).alias('phase'),C('policy_n_hands').alias('reference_n'),*[C(c).alias('reference_'+c) for c in cols if c!='policy_n_hands']);z=f.join(ref,on=['pair_id','phase'],how='left').with_columns(C('reference_n').fill_null(0))
 expr=[]
 for c in nc:
  rr=C('reference_'+c).fill_null(0)
  if c.endswith('_z'):diff=(C(c)/C('policy_n_hands').sqrt()-rr/C('reference_n').clip(1).sqrt())*(C('policy_n_hands')*C('reference_n')/(C('policy_n_hands')+C('reference_n')).clip(0)).sqrt()
  else:diff=C(c)-rr
  expr.append(diff.alias('change_'+c))
 z=z.with_columns(expr).select('pair_id','phase','table_id',*cols,'reference_n',*['change_'+c for c in nc]);allparts.append(z)
f=pl.concat(allparts);f.write_parquet(dest/'features.parquet',compression='zstd');d=f.filter(C('phase')=='development').join(labs,on='pair_id').sort('pair_id');y=d['label'].to_numpy();family=d['behavior_family'].to_numpy();g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());X=d.select(nc).to_numpy();D=d.select(['change_'+c for c in nc]).to_numpy();pred={n:np.zeros(len(d)) for n in ['current','change','aligned_min','aligned_geometric']};params=[]
for fold in folds:
 va=np.isin(g,fold['valid_tables']);tr=~va&(y==0);med=np.median(X[tr],axis=0);scale=np.maximum(np.quantile(X[tr],.9,axis=0)-np.quantile(X[tr],.1,axis=0),.01);dm=np.median(D[tr],axis=0);ds=np.maximum(np.quantile(D[tr],.9,axis=0)-np.quantile(D[tr],.1,axis=0),.01);z=(X-med)/scale;dz=(D-dm)/ds;aligned=z*dz>0
 vals={'current':np.abs(z),'change':np.abs(dz),'aligned_min':np.where(aligned,np.minimum(np.abs(z),np.abs(dz)),0),'aligned_geometric':np.where(aligned,np.sqrt(np.abs(z*dz)),0)}
 for name,v in vals.items():pred[name][va]=np.sort(np.log1p(v[va]),axis=1)[:,-3:].mean(1)
 params.append({'median':med.tolist(),'scale':scale.tolist(),'change_median':dm.tolist(),'change_scale':ds.tolist()})
reports=[]
for method,p in pred.items():
 for b in ['all','directed_transfer','soft_play','coordinated_isolation']:
  keep=np.ones(len(d),bool) if b=='all' else (y==0)|(family==b);reports.append({'method':method,'family':b,'weighted_AP':ap(y[keep],p[keep],sample_weight=np.where(y[keep]>0,1,50))})
print('NOVELTY',reports,flush=True);(dest/'novelty_metrics.json').write_text(json.dumps(reports,indent=2));(dest/'novelty_parameters.json').write_text(json.dumps({'columns':nc,'folds':params}));pd.DataFrame(pred).assign(pair_id=d['pair_id'].to_list(),truth=y).to_csv(dest/'novelty_oof.csv',index=False)
features=cols+['reference_n']+['change_'+c for c in nc];XX=d.select(features).to_numpy();names=['none','directed_transfer','soft_play','coordinated_isolation'];yy=np.array([names.index(b) for b in family]);oof=np.zeros((len(d),4));t=time.time()
for fold in folds:
 va=np.isin(g,fold['valid_tables']);tr=~va;m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=6,random_seed=991+fold['fold'],verbose=False,allow_writing_files=False);m.fit(XX[tr],yy[tr]);oof[va]=m.predict_proba(XX[va]);m.save_model(str(dest/f'fold{fold["fold"]}.cbm'));print('fit',fold['fold'],round(time.time()-t,1),flush=True)
pd.DataFrame(oof,columns=names).assign(pair_id=d['pair_id'].to_list(),truth=yy).to_csv(dest/'oof.csv',index=False);(dest/'columns.json').write_text(json.dumps(features));old=pd.read_csv(root/'residual_oof.csv').sort_values('pair_id');reports=[]
for alpha in [0,.25,.5,.75,1]:
 p=alpha*oof+(1-alpha)*old[names].values;risk=1-p[:,0];reports.append({'contrast_weight':alpha,'AP':ap(y,risk),'weighted_AP':ap(y,risk,sample_weight=np.where(y>0,1,50))})
print('MODEL',reports,flush=True);(dest/'metrics.json').write_text(json.dumps(reports,indent=2))
