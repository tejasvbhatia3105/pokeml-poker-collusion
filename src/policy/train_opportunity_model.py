import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'opportunity_model';dest.mkdir(exist_ok=True);C=pl.col
core=json.loads((root/'residual_columns.json').read_text());labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');parts=[];t=time.time()
for window in ['full','first_2000','last_2000']:
    folder=root if window=='full' else root/'full_window_stress'/window
    h=pl.scan_parquet(str(folder/'hand_features/*.parquet')).filter(C('phase')=='development').join(labs.select('pair_id').lazy(),on='pair_id').collect();vc=[c for c in h.columns if c.endswith('_v')];expr=[]
    for c in vc:
        expr += [C(c).sum().alias('opportunity_'+c+'_sum'),(C(c)>0).sum().alias('opportunity_'+c+'_hands'),C(c).max().alias('opportunity_'+c+'_max'),C(c).top_k(5).mean().alias('opportunity_'+c+'_top5')]
    extra=h.group_by('pair_id').agg(expr).with_columns(pl.selectors.float().cast(pl.Float32));extra.write_parquet(dest/f'{window}_features.parquet')
    d=pl.scan_parquet(str(folder/'pair_features/*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').collect().join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible')).join(extra,on='pair_id').with_columns(pl.lit(window).alias('window'));parts.append(d)
d=pl.concat(parts).with_columns((1/pl.len().over('pair_id')).alias('sample_weight'));cols=core+[c for c in d.columns if c.startswith('opportunity_')];X=d.select(cols).to_numpy();names=['none','directed_transfer','soft_play','coordinated_isolation'];y=np.array([names.index(b) for b in d['behavior_family']]);g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());p=np.zeros((len(d),4));old=np.zeros_like(p);control=np.zeros_like(p);models=[]
for f in folds:
    va=np.isin(g,f['valid_tables']);tr=~va;m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=4,random_seed=991+f['fold'],verbose=False,allow_writing_files=False)
    m.fit(X[tr],y[tr],sample_weight=d['sample_weight'].to_numpy()[tr]);p[va]=m.predict_proba(X[va]);m.save_model(str(dest/f'fold{f["fold"]}.cbm'));models.append(m)
    for path,out in [(root/f'residual_fold{f["fold"]}.cbm',old),(root/f'timed_episodes/control_fold{f["fold"]}.cbm',control)]:
        m=CatBoostClassifier();m.load_model(str(path));out[va]=m.predict_proba(d.filter(pl.Series(va)).select(core).to_numpy())
    print('fit',f['fold'],round(time.time()-t,1),flush=True)
(dest/'columns.json').write_text(json.dumps(cols));pd.DataFrame(p,columns=names).assign(pair_id=d['pair_id'].to_list(),window=d['window'].to_list(),truth=y).to_csv(dest/'oof.csv',index=False);reports=[]
for name,pred in [('v4',old),('matched_control',control),('opportunity',p),('half_blend',.5*(p+old))]:
    for window in ['full','first_2000','last_2000']:
        keep=d['window'].to_numpy()==window;yy=y[keep];risk=1-pred[keep,0];w=np.where(yy>0,1,50);reports.append(dict(model=name,window=window,AP=ap(yy>0,risk),weighted_AP=ap(yy>0,risk,sample_weight=w)))
(dest/'metrics.json').write_text(json.dumps(reports,indent=2));pd.DataFrame(dict(feature=cols,importance=np.mean([m.feature_importances_ for m in models],axis=0))).sort_values('importance',ascending=False).to_csv(dest/'importance.csv',index=False);print('RESULT',reports,flush=True)
