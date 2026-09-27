\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
C=pl.col;root=Path('artifacts/policy');t=time.time()
labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');ev=pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id')
h=pl.scan_parquet(str(root/'hand_features/*.parquet')).filter(C('phase')=='development').join(labs.lazy().select('pair_id'),on='pair_id').collect()
base=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').collect().sort('pair_id');cols=json.loads((root/'residual_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text());models=[]
for k in range(4):
 m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{k}.cbm'));models.append(m)
channels=[c[:-2] for c in h.columns if c.endswith('_r')]
def aggregate(h):
 h=h.sort('pair_id','phase','time_index');keys=['pair_id','phase','player_1','player_2'];stats=[pl.len().alias('policy_n_hands'),C('surprise_max').mean().alias('policy_surprise_mean'),C('surprise_max').top_k(5).mean().alias('policy_surprise_top5')]
 for n in channels:
  r=C(n+'_r');v=C(n+'_v');hz=r/(v+1).sqrt()
  stats += [(r.sum()/(v.sum()+1).sqrt()).alias(n+'_z'),hz.mean().alias(n+'_mean'),hz.std().alias(n+'_std'),hz.top_k(5).mean().alias(n+'_top5'),(-hz).top_k(5).mean().alias(n+'_bottom5')]
 f=h.group_by(keys).agg(stats);roll=[]
 for w in [5,10,20]:
  for n in channels:roll.append((C(n+'_r').rolling_sum(w,min_samples=1).over(['pair_id','phase'])/(C(n+'_v').rolling_sum(w,min_samples=1).over(['pair_id','phase'])+1).sqrt()).alias(n+f'_w{w}'))
 rh=h.with_columns(roll);f=f.join(rh.group_by(keys).agg(*[e for w in [5,10,20] for n in channels for e in [C(n+f'_w{w}').max().alias(n+f'_w{w}_max'),C(n+f'_w{w}').min().alias(n+f'_w{w}_min')]]),on=keys)
 return f.with_columns(pl.selectors.float().cast(pl.Float32)).fill_null(0).join(base.select('pair_id','table_id','label','behavior_family'),on='pair_id').sort('pair_id')
def predict(d):
 x=d.select(cols).to_numpy();fold=np.array([folds[t] for t in d['table_id']]);p=np.zeros((len(d),4))
 for k,m in enumerate(models):p[fold==k]=m.predict_proba(x[fold==k],thread_count=4)
 return p
reports=[];baseline=predict(base)
for name,mask in [('full',pl.lit(True)),('first_2000',C('time_index')<2000),('last_2000',C('time_index')>=1000)]:
 sh=h.filter(mask);d=aggregate(sh);d.write_parquet(root/f'exposure_{name}.parquet');p=predict(d)
 supported=sh.select('pair_id','hand_id').join(ev,on=['pair_id','hand_id']).select('pair_id').unique()['pair_id'].to_list();keep=(d['label']==0).to_numpy()|d['pair_id'].is_in(supported).to_numpy();y=d['label'].to_numpy();d.select('pair_id').with_columns(pl.Series('eligible',keep)).write_parquet(root/f'exposure_{name}_eligibility.parquet');risk=1-p[:,0];r={'window':name,'rows':len(d),'scored_negatives':int(((y==0)&keep).sum()),'scored_supported_positives':int(((y==1)&keep).sum()),'unscored_positives_without_retained_evidence':int(((y==1)&~keep).sum()),'AP':average_precision_score(y[keep],risk[keep]),'AP_negative_weight50':average_precision_score(y[keep],risk[keep],sample_weight=np.where(y[keep]>0,1,50))}
 if name=='full':
  assert d['pair_id'].to_list()==base['pair_id'].to_list()
  r['max_feature_reconstruction_error']=float(np.max(np.abs(d.select(cols).to_numpy()-base.select(cols).to_numpy())))
  r['max_prediction_reconstruction_error']=float(np.max(np.abs(p-baseline)))
 else:
                                                                       
  ix={pid:i for i,pid in enumerate(base['pair_id'])};br=np.array([1-baseline[ix[pid],0] for pid in d['pair_id']])
  r['matched_full_period_AP_negative_weight50']=average_precision_score(y[keep],br[keep],sample_weight=np.where(y[keep]>0,1,50))
 r['family_weighted_AP']={b:average_precision_score(y[keep&((d['behavior_family']==b).to_numpy()|(y==0))],risk[keep&((d['behavior_family']==b).to_numpy()|(y==0))],sample_weight=np.where(y[keep&((d['behavior_family']==b).to_numpy()|(y==0))]>0,1,50)) for b in ['directed_transfer','soft_play','coordinated_isolation']}
 reports.append(r);print(r,'seconds',round(time.time()-t,1),flush=True)
report={'results':reports,'limitations':['No evaluation labels used.','Only aggregation exposure is truncated; cached residuals still use full-development-phase player-style estimates.','Positive eligibility requires a retained public evidence hand; targets without one are excluded, not relabelled negative.','This is a robustness audit on existing validation pools, not a new untouched holdout.']};(root/'exposure_stress.json').write_text(json.dumps(report,indent=2))
