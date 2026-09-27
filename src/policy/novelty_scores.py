import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,polars as pl
root=Path('artifacts/policy');C=pl.col
labs=pl.read_csv('data/development_labels.csv')
d=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(C('phase')=='development').join(labs.lazy().select('pair_id','label','behavior_family'),on='pair_id').collect().sort('pair_id')
cols=[c for c in d.columns if c.endswith(('_z','_w10_max','_w10_min'))];X=d.select(cols).to_numpy();y=d['label'].to_numpy();g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());oof=np.zeros(len(d));params=[]
for f in folds:
 va=np.isin(g,f['valid_tables']);tr=~va&(y==0);med=np.median(X[tr],axis=0);scale=np.maximum(np.quantile(X[tr],.9,axis=0)-np.quantile(X[tr],.1,axis=0),.01)
 z=np.log1p(np.abs((X-med)/scale));oof[va]=np.sort(z[va],axis=1)[:,-3:].mean(1);params.append((med,scale))
np.savez(root/'novelty_parameters.npz',med=np.array([p[0] for p in params]),scale=np.array([p[1] for p in params]),columns=np.array(cols))
d.select('pair_id','table_id','behavior_family','label').with_columns(pl.Series('novelty',oof)).write_csv(root/'novelty_oof.csv')
parts=[]
for path in sorted((root/'pair_features').glob('*.parquet')):
 e=pl.read_parquet(path);ex=e.select(cols).to_numpy();z=np.mean([np.log1p(np.abs((ex-med)/scale)) for med,scale in params],axis=0)
 score=np.sort(z,axis=1)[:,-3:].mean(1);top=np.argmax(z,axis=1)
 parts.append(e.select('pair_id','phase','table_id','player_1','player_2','policy_n_hands').with_columns(pl.Series('novelty',score),pl.Series('top_channel',[cols[k] for k in top])))
e=pl.concat(parts);e.write_parquet(root/'novelty_all.parquet')
ev=pl.read_csv('artifacts/selected_pair_eval.csv');e.filter(C('phase')=='evaluation').join(ev,on='pair_id').with_columns((1-C('none')).alias('v2_risk')).sort('novelty',descending=True).write_csv(root/'novelty_eval_audit.csv')
print('normal novelty quantiles',np.quantile(oof[y==0],[.5,.9,.99,1]));print('positive novelty quantiles',np.quantile(oof[y==1],[0,.1,.5,.9,1]))
