import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'timed_episodes';dest.mkdir(exist_ok=True);C=pl.col
core=json.loads((root/'residual_columns.json').read_text());labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');parts=[];t0=time.time()
for window in ['full','first_2000','last_2000']:
    folder=root if window=='full' else root/'full_window_stress'/window
    h=pl.scan_parquet(str(folder/'hand_features/*.parquet')).filter(C('phase')=='development').join(labs.select('pair_id').lazy(),on='pair_id').collect().sort('pair_id','time_index');channels=[c[:-2] for c in h.columns if c.endswith('_r')];rows=[]
    for key,g in h.partition_by('pair_id',as_dict=True).items():
        times=g['time_index'].to_numpy();r=g.select([c+'_r' for c in channels]).to_numpy().astype(float);v=g.select([c+'_v' for c in channels]).to_numpy().astype(float);cr=np.vstack([np.zeros(len(channels)),np.cumsum(r,axis=0)]);cv=np.vstack([np.zeros(len(channels)),np.cumsum(v,axis=0)]);right=np.arange(1,len(g)+1);row=dict(pair_id=key[0])
        for span in [25,100,400]:
            left=np.searchsorted(times,times-span,side='right');z=(cr[right]-cr[left])/np.sqrt(cv[right]-cv[left]+1);eligible=(right-left)>=2
            for j,c in enumerate(channels):
                row[f'timed_{span}_{c}_max']=float(max(0,z[eligible,j].max())) if eligible.any() else 0.
                row[f'timed_{span}_{c}_min']=float(min(0,z[eligible,j].min())) if eligible.any() else 0.
        rows.append(row)
    extra=pl.DataFrame(rows).with_columns(pl.selectors.float().cast(pl.Float32));extra.write_parquet(dest/f'{window}_features.parquet')
    d=pl.scan_parquet(str(folder/'pair_features/*.parquet')).filter(C('phase')=='development').join(labs.lazy(),on='pair_id').collect().join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible')).join(extra,on='pair_id').with_columns(pl.lit(window).alias('window'));parts.append(d)
    print('features',window,len(d),round(time.time()-t0,1),flush=True)
d=pl.concat(parts).with_columns((1/pl.len().over('pair_id')).alias('sample_weight'));names=['none','directed_transfer','soft_play','coordinated_isolation'];y=np.array([names.index(b) for b in d['behavior_family']]);g=d['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());weights=d['sample_weight'].to_numpy();reports=[];base=np.zeros((len(d),4))
for f in folds:
    va=np.isin(g,f['valid_tables']);m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));base[va]=m.predict_proba(d.filter(pl.Series(va)).select(core).to_numpy())
for mode in ['control','timed']:
    cols=core+([c for c in d.columns if c.startswith('timed_')] if mode=='timed' else []);X=d.select(cols).to_numpy();p=np.zeros_like(base)
    for f in folds:
        va=np.isin(g,f['valid_tables']);tr=~va;m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=4,random_seed=991+f['fold'],verbose=False,allow_writing_files=False)
        m.fit(X[tr],y[tr],sample_weight=weights[tr]);p[va]=m.predict_proba(X[va]);m.save_model(str(dest/f'{mode}_fold{f["fold"]}.cbm'));print('fit',mode,f['fold'],round(time.time()-t0,1),flush=True)
    (dest/f'{mode}_columns.json').write_text(json.dumps(cols));pd.DataFrame(p,columns=names).assign(pair_id=d['pair_id'].to_list(),window=d['window'].to_list(),truth=y).to_csv(dest/f'{mode}_oof.csv',index=False)
    for alpha in [0,.5,1]:
        pred=alpha*p+(1-alpha)*base
        for window in ['full','first_2000','last_2000']:
            keep=d['window'].to_numpy()==window;yy=y[keep];risk=1-pred[keep,0];behavior=pred[keep,1:].argmax(1)+1;w=np.where(yy>0,1,50)
            reports.append(dict(mode=mode,blend=alpha,window=window,AP=ap(yy>0,risk),weighted_AP=ap(yy>0,risk,sample_weight=w),weighted_behavior_AP=float(np.mean([ap(yy==k,risk*(behavior==k),sample_weight=w) for k in [1,2,3]]))))
    (dest/'metrics.json').write_text(json.dumps(reports,indent=2));print('RESULT',mode,[r for r in reports if r['mode']==mode],flush=True)
