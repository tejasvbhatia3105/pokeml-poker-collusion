import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,itertools
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
C=pl.col;root=Path('artifacts/policy');dest=root/'pair_features';hdest=root/'hand_features'
HANDS_ALL=os.environ.get('HAND_ROWS_ALL')                                                                                  
window=os.environ.get('POLICY_WINDOW')
if window:
 from window_actions import window_actions
 if window not in ['first_2000','last_2000'] and not window.startswith('w'):raise ValueError(window)
 dest=root/'full_window_stress'/window/'pair_features';hdest=root/'full_window_stress'/window/'hand_features'
dest.mkdir(exist_ok=True,parents=True);hdest.mkdir(exist_ok=True,parents=True)
cols=json.loads((root/'feature_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text())
models=[];sizes=[]
for f in range(4):
 m=CatBoostClassifier();m.load_model(str(root/f'action_fold{f}.cbm'));models.append(m)
 m=CatBoostRegressor();m.load_model(str(root/f'size_fold{f}.cbm'));sizes.append(m)
known=pl.concat([pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2')])
t=time.time()
for ti,path in enumerate(sorted((root/'actions').glob('*.parquet'))):
 table=path.stem;out=dest/path.name
 if HANDS_ALL:
  out=Path(HANDS_ALL)/path.name;Path(HANDS_ALL).mkdir(exist_ok=True,parents=True)
 if out.exists():continue
 a=pl.read_parquet(path)
 if window:a=window_actions(a,window)
 X=a.select(cols).to_numpy();dev=a['phase'].to_numpy()=='development';act=a['action_class'].to_numpy();p=np.zeros((len(a),4));sz=np.zeros(len(a));agg=act==3
 for mask,fs in [(dev,[folds[table]]),(~dev,range(4))]:
  if not mask.any():continue
  p[mask]=np.mean([models[f].predict_proba(X[mask],thread_count=4) for f in fs],axis=0)
  sm=mask&agg
  if sm.any():sz[sm]=np.mean([sizes[f].predict(X[sm],thread_count=4) for f in fs],axis=0)
                                                                                        
 call=a['to_call'].to_numpy()>0
 legal=np.ones_like(p);legal[call,1]=0;legal[~call,0]=0;legal[~call,2]=0
 invalid=legal[np.arange(len(a)),act]==0
 legal[invalid]=1;p*=legal;p/=p.sum(1,keepdims=True)
 resid=np.eye(4)[act]-p;v=p*(1-p)
 a=a.with_columns(*[pl.Series(f'r{k}',resid[:,k]) for k in range(4)],*[pl.Series(f'v{k}',v[:,k]) for k in range(4)],pl.Series('surprise',-np.log(p[np.arange(len(a)),act].clip(1e-7))),pl.Series('size_r',np.where(agg,a['log_bet_ratio'].to_numpy()-sz,0)))
                                                                                         
 a=a.with_columns(*[(C(f'r{k}')-C(f'r{k}').mean().over(['player_id','phase','street_no'])).alias(f'r{k}') for k in range(4)],C('street_no').cast(pl.Int64))
 st=pl.read_parquet(root/'states'/path.name).select('hand_id','street_no',C('player_id').alias('partner'),C('fold_no').alias('partner_fold'),C('equity').alias('partner_eq'))
 z=a.join(st,on=['hand_id','street_no']).filter(C('player_id')!=C('partner'))
 z=z.with_columns(pl.min_horizontal('player_id','partner').alias('player_1'),pl.max_horizontal('player_id','partner').alias('player_2'),(C('partner_fold')>=C('action_no')).alias('alive'),(C('last_aggressor')==C('partner')).fill_null(False).alias('facing'))
 alive=C('alive');facing=C('facing')&alive;hu=alive&(C('players_active')==2);out3=alive&~facing&(C('players_active')>=3);weak=C('equity')<.45;strong=C('equity')>.65;hidden=C('partner_eq')-.5
                                                                                                       
 channels={
 'alive_agg':(3,alive,1),'alive_fold':(0,alive,1),
 'partner_call':(2,facing,1),'weak_partner_call':(2,facing&weak,1),
 'partner_surrender':(0,facing&strong,1),'hu_passivity':(3,hu&strong,-1),
 'hu_check':(1,hu&strong,1),'outsider_agg':(3,out3,1),'weak_outsider_agg':(3,out3&weak,1),
 'dealt_weak_agg':(3,weak,1),'hidden_agg_alive':(3,alive,hidden),'hidden_fold_alive':(0,alive,hidden),
 'hidden_agg_folded':(3,~alive,hidden),'hidden_call_folded':(2,~alive,hidden),
 'yield_better':(0,facing&(C('equity')>C('partner_eq')+.15),1),
 }
 expr=[]
 for name,(k,mask,coef) in channels.items():
  expr += [pl.when(mask).then(C(f'r{k}')*coef).otherwise(0).alias(name+'_r'),pl.when(mask).then((C(f'v{k}')+.01)*coef*coef).otherwise(0).alias(name+'_v')]
 for name,mask in [('size_partner',facing),('size_outsider',out3)]:
  expr += [pl.when(mask&(C('action_class')==3)).then(C('size_r')).otherwise(0).alias(name+'_r'),(mask&(C('action_class')==3)).cast(pl.Float64).alias(name+'_v')]
  channels[name]=None
 z=z.with_columns(expr)
                                                                                     
 roster=pl.read_parquet(root/'states'/path.name).select('hand_id',C('player_id').alias('player_1')).unique()
 base=roster.join(roster.rename({'player_1':'player_2'}),on='hand_id').filter(C('player_1')<C('player_2')).join(a.select('hand_id','phase','time_index').unique(),on='hand_id')
 extra=[]
 if HANDS_ALL:
  is1=C('player_id')==C('player_1')
  extra=[C('surprise').filter(is1).sum().alias('surp_sum_1'),C('surprise').filter(~is1).sum().alias('surp_sum_2'),C('surprise').filter(is1&alive).sum().alias('surp_alive_1'),C('surprise').filter(~is1&alive).sum().alias('surp_alive_2'),C('surprise').filter(is1&facing).sum().alias('surp_facing_1'),C('surprise').filter(~is1&facing).sum().alias('surp_facing_2'),is1.sum().alias('n_act_1'),(~is1).sum().alias('n_act_2'),(is1&facing).sum().alias('n_facing_1'),(~is1&facing).sum().alias('n_facing_2'),C('surprise').filter(is1).max().alias('surp_max_1'),C('surprise').filter(~is1).max().alias('surp_max_2'),(C('surprise').filter(alive)).sum().alias('surp_alive_sum')]
 h=z.group_by('hand_id','player_1','player_2').agg(*[C(n+s).sum() for n in channels for s in ['_r','_v']],C('surprise').filter(alive).max().alias('surprise_max'),*extra)
 h=base.join(h,on=['hand_id','player_1','player_2'],how='left').fill_null(0).join(known,on=['player_1','player_2'],how='left').with_columns(C('pair_id').fill_null(C('player_1')+'_'+C('player_2'))).sort('pair_id','phase','time_index')
 if HANDS_ALL:
  h.with_columns(pl.selectors.float().cast(pl.Float32)).write_parquet(out,compression='zstd')
  if ti%40==0:print(ti,table,'hand rows',len(h),round(time.time()-t,1),flush=True)
  continue
 keys=['pair_id','phase','player_1','player_2'];stats=[pl.len().alias('policy_n_hands'),C('surprise_max').mean().alias('policy_surprise_mean'),C('surprise_max').top_k(5).mean().alias('policy_surprise_top5')]
 for n in channels:
  r=C(n+'_r');v=C(n+'_v');hz=r/(v+1).sqrt()
  stats += [(r.sum()/(v.sum()+1).sqrt()).alias(n+'_z'),hz.mean().alias(n+'_mean'),hz.std().alias(n+'_std'),hz.top_k(5).mean().alias(n+'_top5'),(-hz).top_k(5).mean().alias(n+'_bottom5')]
 f=h.group_by(keys).agg(stats)
                                                                  
 roll=[]
 for w in [5,10,20]:
  for n in channels:
   roll.append((C(n+'_r').rolling_sum(w,min_samples=1).over(['pair_id','phase'])/(C(n+'_v').rolling_sum(w,min_samples=1).over(['pair_id','phase'])+1).sqrt()).alias(n+f'_w{w}'))
 rh=h.with_columns(roll)
 f=f.join(rh.group_by(keys).agg(*[e for w in [5,10,20] for n in channels for e in [C(n+f'_w{w}').max().alias(n+f'_w{w}_max'),C(n+f'_w{w}').min().alias(n+f'_w{w}_min')]]),on=keys)
 f=f.with_columns(pl.lit(table).alias('table_id'),pl.selectors.float().cast(pl.Float32)).fill_null(0)
 f.write_parquet(out,compression='zstd')
                                                               
 hand_keep=pl.read_csv('data/development_labels.csv').select('pair_id') if window else known.select('pair_id')
 h.join(hand_keep,on='pair_id',how='semi').with_columns(pl.selectors.float().cast(pl.Float32)).write_parquet(hdest/path.name,compression='zstd')
 if ti%10==0:print(ti,table,'seconds',round(time.time()-t,1),'invalid_actions',int(invalid.sum()),'pairs',len(f),flush=True)
 if os.environ.get('ONE_TABLE'):break
