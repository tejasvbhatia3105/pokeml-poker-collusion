import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'value_model';dest.mkdir(exist_ok=True);C=pl.col
while len(list((root/'value_pairs').glob('*.parquet')))<400:time.sleep(3)
labs=pl.read_csv('data/development_labels.csv').select('pair_id');vh=pl.scan_parquet(str(root/'value_hands/*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').collect();raw=[c for c in vh.columns if c.startswith('value_')];parts=[]
for name,mask in [('full',pl.lit(True)),('first_2000',C('time_index')<2000),('last_2000',C('time_index')>=1000)]:
 stats=[]
 for n in raw:
  x=C(n);stats += [x.mean().alias(n+'_mean'),x.max().alias(n+'_max'),x.top_k(3).mean().alias(n+'_top3'),x.sum().alias(n+'_sum'),(x>5).mean().alias(n+'_over5_rate'),(x+1).log().mean().alias(n+'_logmean')]
 values=vh.filter(mask).group_by('pair_id').agg(stats)
 d=pl.read_parquet(root/f'exposure_{name}.parquet').join(pl.read_parquet(root/f'exposure_{name}_eligibility.parquet'),on='pair_id').filter(C('eligible')).join(values,on='pair_id',how='left').fill_null(0).with_columns(pl.lit(name).alias('window'));parts.append(d)
d=pl.concat(parts).with_columns((1/pl.len().over('pair_id')).alias('sample_weight'));oldcols=json.loads((root/'residual_columns.json').read_text());newcols=[c for c in d.columns if c.startswith('value_')];cols=oldcols+newcols;X=d.select(cols).to_numpy();names=['none','directed_transfer','soft_play','coordinated_isolation'];y=np.array([names.index(b) for b in d['behavior_family']]);g=d['table_id'].to_numpy();w=d['sample_weight'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());oof=np.zeros((len(d),4));old=np.zeros_like(oof);models=[];t=time.time()
for f in folds:
 va=np.isin(g,f['valid_tables']);tr=~va;m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=6,random_seed=991+f['fold'],verbose=False,allow_writing_files=False);m.fit(X[tr],y[tr],sample_weight=w[tr]);oof[va]=m.predict_proba(X[va]);m.save_model(str(dest/f'fold{f["fold"]}.cbm'));models.append(m)
 base=CatBoostClassifier();base.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));old[va]=base.predict_proba(d.filter(pl.Series(va)).select(oldcols).to_numpy());print('fold',f['fold'],round(time.time()-t,1),flush=True)
pd.DataFrame(oof,columns=names).assign(pair_id=d['pair_id'].to_list(),window=d['window'].to_list(),truth=y).to_csv(dest/'oof.csv',index=False);(dest/'columns.json').write_text(json.dumps(cols));reports=[]
for alpha in [0,.25,.5,.75,1]:
 p=alpha*oof+(1-alpha)*old
 for window in ['full','first_2000','last_2000']:
  keep=(d['window']==window).to_numpy();yy=y[keep];risk=1-p[keep,0];pred=p[keep,1:].argmax(1)+1;weights=np.where(yy>0,1,50);reports.append({'value_weight':alpha,'window':window,'AP':ap(yy>0,risk),'weighted_AP':ap(yy>0,risk,sample_weight=weights),'weighted_behavior_AP':np.mean([ap(yy==k,risk*(pred==k),sample_weight=weights) for k in [1,2,3]])})
print(reports,flush=True);(dest/'metrics.json').write_text(json.dumps(reports,indent=2));pd.DataFrame({'feature':cols,'importance':np.mean([m.feature_importances_ for m in models],0)}).sort_values('importance',ascending=False).to_csv(dest/'importance.csv',index=False)
