import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl
from catboost import CatBoostClassifier
C=pl.col;root=Path('artifacts/policy');dest=root/'joint_pairs';dest.mkdir(exist_ok=True);cols=json.loads((root/'feature_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text());models=[]
for f in range(4):
 m=CatBoostClassifier();m.load_model(str(root/f'action_fold{f}.cbm'));models.append(m)
t=time.time()
for ti,path in enumerate(sorted((root/'actions').glob('*.parquet'))):
 out=dest/path.name
 if out.exists():continue
 a=pl.read_parquet(path);X=a.select(cols).to_numpy();dev=a['phase'].to_numpy()=='development';act=a['action_class'].to_numpy();p=np.zeros((len(a),4))
 for mask,fs in [(dev,[folds[path.stem]]),(~dev,range(4))]:p[mask]=np.mean([models[f].predict_proba(X[mask],thread_count=4) for f in fs],axis=0)
 call=a['to_call'].to_numpy()>0;legal=np.ones_like(p);legal[call,1]=0;legal[~call,0]=0;legal[~call,2]=0;assert np.all(legal[np.arange(len(a)),act]>0);p*=legal;p/=p.sum(1,keepdims=True);resid=np.eye(4)[act]-p
 a=a.with_columns(*[pl.Series(f'r{k}',resid[:,k]) for k in [0,2,3]],*[pl.Series(f'v{k}',p[:,k]*(1-p[:,k])) for k in [0,2,3]])
 a=a.with_columns(*[(C(f'r{k}')-C(f'r{k}').mean().over(['player_id','phase','street_no'])).alias(f'r{k}') for k in [0,2,3]])
 expr=[]
 for st,mask in [('all',pl.lit(True)),('pre',C('street_no')==0),('post',C('street_no')>0)]:
  for k in [0,2,3]:expr.append((C(f'r{k}').filter(mask).sum()/(C(f'v{k}').filter(mask).sum()+1).sqrt()).alias(f'{st}_{k}'))
 own=a.group_by('hand_id','player_id').agg(expr)
 roster=pl.read_parquet(root/'states'/path.name).select('hand_id','player_id').unique().join(own,on=['hand_id','player_id'],how='left').fill_null(0)
 z=roster.rename({'player_id':'player_1'}).join(roster.rename({'player_id':'player_2',**{c:c+'_b' for c in own.columns if c not in ['hand_id','player_id']}}),on='hand_id').filter(C('player_1')<C('player_2')).join(a.select('hand_id','phase','time_index').unique(),on='hand_id')
 pairs=pl.read_parquet(root/'pair_features'/path.name).select('player_1','player_2','pair_id').unique();z=z.join(pairs,on=['player_1','player_2']).sort('pair_id','phase','time_index');expr=[];features=[]
 for st in ['all','pre','post']:
  for k,n in [(0,'fold'),(2,'call'),(3,'agg')]:
   name=f'joint_{st}_{n}_sync';features.append(name);expr.append((C(f'{st}_{k}')*C(f'{st}_{k}_b')).alias(name))
   name=f'joint_{st}_{n}_direction';features.append(name);expr.append((C(f'{st}_{k}')-C(f'{st}_{k}_b')).alias(name))
  for k,j,n in [(3,0,'agg_fold'),(3,2,'agg_call'),(2,0,'call_fold')]:
   name=f'joint_{st}_{n}_exchange';features.append(name);expr.append((C(f'{st}_{k}')*C(f'{st}_{j}_b')+C(f'{st}_{j}')*C(f'{st}_{k}_b')).alias(name))
 z=z.with_columns(expr);keys=['pair_id','phase'];stats=[]
 for n in features:
  x=C(n);stats += [x.mean().alias(n+'_mean'),x.std().alias(n+'_std'),x.top_k(5).mean().alias(n+'_top5'),(-x).top_k(5).mean().alias(n+'_bottom5'),x.abs().mean().alias(n+'_absmean')]
 outdata=z.group_by(keys).agg(stats);roll=[C(n).rolling_mean(w,min_samples=1).over(keys).abs().alias(n+f'_w{w}') for n in features for w in [5,10]]
 outdata=outdata.join(z.with_columns(roll).group_by(keys).agg(*[C(n+f'_w{w}').max() for n in features for w in [5,10]]),on=keys).with_columns(pl.lit(path.stem).alias('table_id'),pl.selectors.float().cast(pl.Float32)).fill_null(0)
 outdata.write_parquet(out,compression='zstd')
 if ti%40==0:print(ti,'seconds',round(time.time()-t,1),flush=True)
