import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,polars as pl
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score
channels=['weak_call_partner','weak_call_hu','weak_agg_out','strong_check_hu','strong_fold_partner','strong_call_partner','both_showdown','act','agg_partner','agg_out','team_net','min_contrib','weak_call_partner_money','strong_fold_partner_maxcall','strong_fold_out','both_survive','check_hu','preflop_agg_partner','preflop_fold_partner','preflop_call_partner','preflop_weak_agg_out']
cols=[n+'_mean_excess' for n in channels]
d=pl.scan_parquet('artifacts/context_pairs/*.parquet').filter(pl.col('phase')=='development').select('pair_id','table_id','n_hands',*cols).join(pl.read_csv('data/development_labels.csv').lazy().select('pair_id','label','behavior_family'),on='pair_id').collect().sort('pair_id')
X=d.select(cols).to_numpy()*np.sqrt(d['n_hands'].to_numpy()[:,None]);y=d['label'].to_numpy();b=d['behavior_family'].to_numpy();g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());pred={k:np.zeros(len(d)) for k in ['mahalanobis','max','top3']}
for f in folds:
 va=np.isin(g,f['valid_tables']);tr=~va&(y==0)
 med=np.median(X[tr],axis=0);scale=np.quantile(X[tr],.9,axis=0)-np.quantile(X[tr],.1,axis=0);scale=np.maximum(scale,1e-5)
 z=(X-med)/scale;z=np.sign(z)*np.log1p(np.abs(z));m=LedoitWolf().fit(z[tr]);pred['mahalanobis'][va]=m.mahalanobis(z[va]);pred['max'][va]=np.max(np.abs(z[va]),axis=1);pred['top3'][va]=np.sort(np.abs(z[va]),axis=1)[:,-3:].mean(axis=1)
r=[]
for n,p in pred.items():
 for family in ['all','directed_transfer','soft_play','coordinated_isolation']:
  keep=np.ones(len(d),bool) if family=='all' else (b==family)|(y==0)
  r.append({'method':n,'family':family,'AP':average_precision_score(y[keep],p[keep]),'AP_negative_weight50':average_precision_score(y[keep],p[keep],sample_weight=np.where(y[keep]>0,1,50))})
print(json.dumps(r,indent=2));Path('artifacts/normal_deviations_diagnostic.json').write_text(json.dumps(r,indent=2))
