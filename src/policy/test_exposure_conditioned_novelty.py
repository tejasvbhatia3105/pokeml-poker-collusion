\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from catboost import CatBoostClassifier
root=Path('artifacts/policy');duration_matched=os.environ.get('REFERENCE_DURATION')=='matched';dest=root/('exposure_duration_novelty' if duration_matched else 'exposure_conditioned_novelty');dest.mkdir(exist_ok=True);C=pl.col
cols=json.loads((root/'residual_columns.json').read_text());nc=[c for c in cols if c.endswith(('_z','_w10_max','_w10_min'))];labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');folds=json.loads(Path('artifacts/folds.json').read_text())
full=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(C('phase')=='development').collect().join(labs,on='pair_id');tables=full['table_id'].to_numpy();normal=full['label'].to_numpy()==0;NX=full.select(nc).to_numpy();logn=np.log1p(full['policy_n_hands'].to_numpy());scores=[];t=time.time()
for window in ['full','first_2000','last_2000']:
    refdata=full
    if duration_matched and window!='full':
        refdata=pl.concat([pl.scan_parquet(str(root/'full_window_stress'/w/'pair_features/*.parquet')).join(labs.lazy(),on='pair_id').collect() for w in ['first_2000','last_2000']])
    tables=refdata['table_id'].to_numpy();normal=refdata['label'].to_numpy()==0;NX=refdata.select(nc).to_numpy();logn=np.log1p(refdata['policy_n_hands'].to_numpy());neighbor_count=400 if duration_matched and window!='full' else 200
    folder=root/'pair_features' if window=='full' else root/'full_window_stress'/window/'pair_features';a=pl.scan_parquet(str(folder/'*.parquet')).filter(C('phase')=='development').collect().sort('pair_id');output=[]
    for f in folds:
        tr=normal&~np.isin(tables,f['valid_tables']);v=a.filter(C('table_id').is_in(f['valid_tables']));x=v.select(nc).to_numpy();counts=v['policy_n_hands'].to_numpy();values,ix=np.unique(counts,return_inverse=True);ref=NX[tr];rn=logn[tr];med=[];scale=[]
        for count in values:
            neighbors=np.argsort(abs(rn-np.log1p(count)))[:neighbor_count];rows=ref[neighbors];med.append(np.median(rows,axis=0));scale.append(np.maximum(np.quantile(rows,.9,axis=0)-np.quantile(rows,.1,axis=0),.01))
        z=np.log1p(abs((x-np.array(med)[ix])/np.array(scale)[ix]));s=np.sort(z,axis=1)[:,-3:].mean(1);rarity=-np.log1p(-rankdata(s)/(len(s)+1))
        m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));base=1-m.predict_proba(v.select(cols).to_numpy(),thread_count=4)[:,0];br=-np.log1p(-rankdata(base)/(len(base)+1))
        output.append(v.select('pair_id').with_columns(pl.Series('conditional_novelty',rarity),pl.Series('v4_rarity',br)))
    z=pl.concat(output).join(labs,on='pair_id').join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible')).with_columns(pl.lit(window).alias('window'));scores.append(z);print(window,'seconds',round(time.time()-t,1),flush=True)
new=pl.concat(scores).to_pandas();new.to_csv(dest/'scores.csv',index=False);base=pd.read_csv(root/'crossed_validation/oof.csv');base=base[base.training=='full_training'];d=base.merge(new[['pair_id','window','conditional_novelty','v4_rarity']],on=['pair_id','window']);reports=[]
for (family,window),z in d.groupby(['withheld_family','window']):
    yy=z.truth.to_numpy();keep=np.ones(len(z),bool) if family=='all_known' else (yy==0)|(z.behavior.to_numpy()==family);risk=z.v4_rarity.to_numpy() if family=='all_known' else z.rarity.to_numpy()
    for method in ['novelty','conditional_novelty']:
        for weight in [0,.9,.95]:
            p=np.maximum(risk,weight*z[method].to_numpy());reports.append(dict(withheld_family=family,window=window,method=method,weight=weight,weighted_AP=ap(yy[keep],p[keep],sample_weight=np.where(yy[keep]>0,1,50))))
(dest/'metrics.json').write_text(json.dumps(reports,indent=2));print(pd.DataFrame(reports).query('weight == 0.95').pivot_table(index=['window','method'],columns='withheld_family',values='weighted_AP').to_string())
