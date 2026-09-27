\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from scipy.stats import rankdata
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'crossed_validation';dest.mkdir(exist_ok=True);C=pl.col
cols=json.loads((root/'residual_columns.json').read_text());nc=[c for c in cols if c.endswith(('_z','_w10_max','_w10_min'))];labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');folds=json.loads(Path('artifacts/folds.json').read_text());views={};parts=[]
for window in ['full','first_2000','last_2000']:
    folder=root/'pair_features' if window=='full' else root/'full_window_stress'/window/'pair_features'
    a=pl.scan_parquet(str(folder/'*.parquet')).filter(C('phase')=='development').collect().sort('pair_id')
    d=a.join(labs,on='pair_id').join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible')).sort('pair_id').with_columns(pl.lit(window).alias('window'))
    views[window]=dict(frame=a,X=a.select(cols).to_numpy(),NX=a.select(nc).to_numpy(),groups=a['table_id'].to_numpy(),labeled=d)
    parts.append(d)
d=pl.concat(parts).with_columns((1/pl.len().over('pair_id')).alias('sample_weight'));X=d.select(cols).to_numpy();y=d['label'].to_numpy();b=d['behavior_family'].to_numpy();g=d['table_id'].to_numpy();win=d['window'].to_numpy();weights=d['sample_weight'].to_numpy();reports=[];outputs=[];t=time.time()
for mode in ['full_training','augmented_training']:
    for family in ['all_known','directed_transfer','soft_play','coordinated_isolation']:
        pred={view:{name:np.zeros(len(z['labeled'])) for name in ['base','rarity','novelty']} for view,z in views.items()}
        for f in folds:
            train=~np.isin(g,f['valid_tables'])
            if mode=='full_training':train &=win=='full'
            if family!='all_known':train &=b!=family
            m=CatBoostClassifier(iterations=600,depth=4,learning_rate=.045,loss_function='Logloss',l2_leaf_reg=10,verbose=False,thread_count=4,random_seed=318+f['fold'],allow_writing_files=False)
            m.fit(X[train],y[train],sample_weight=weights[train] if mode=='augmented_training' else None)
            m.save_model(str(dest/f'{mode}_{family}_fold{f["fold"]}.cbm'))
            param=np.load(root/f'monotone_novelty/scale_fold{f["fold"]}.npz')
            for view,z in views.items():
                av=np.isin(z['groups'],f['valid_tables']);ld=z['labeled'];va=ld['table_id'].is_in(f['valid_tables']).to_numpy();p=m.predict_proba(z['X'][av],thread_count=4)[:,1];nr=np.sort(np.log1p(abs((z['NX'][av]-param['median'])/param['scale'])),axis=1)[:,-3:].mean(1)
                frame=pd.DataFrame(dict(pair_id=z['frame'].filter(pl.Series(av))['pair_id'].to_list(),base=p,rarity=-np.log1p(-rankdata(p)/(len(p)+1)),novelty=-np.log1p(-rankdata(nr)/(len(nr)+1)))).set_index('pair_id').loc[ld.filter(pl.Series(va))['pair_id'].to_list()]
                for name in pred[view]:pred[view][name][va]=frame[name]
            print(mode,family,f['fold'],round(time.time()-t,1),flush=True)
        for view,z in views.items():
            ld=z['labeled'];yy=ld['label'].to_numpy();bb=ld['behavior_family'].to_numpy();keep=np.ones(len(ld),bool) if family=='all_known' else (yy==0)|(bb==family)
            for alpha in [0,.85,.9,.95]:
                score=np.maximum(pred[view]['rarity'],alpha*pred[view]['novelty']);reports.append(dict(training=mode,withheld_family=family,window=view,novelty_weight=alpha,weighted_AP=ap(yy[keep],score[keep],sample_weight=np.where(yy[keep]>0,1,50))))
            outputs.append(pd.DataFrame(dict(pair_id=ld['pair_id'].to_list(),training=mode,withheld_family=family,window=view,truth=yy,behavior=bb,**pred[view])))
        pd.concat(outputs).to_csv(dest/'oof.csv',index=False);(dest/'metrics.json').write_text(json.dumps(reports,indent=2))
        print('RESULT',mode,family,[r for r in reports if r['training']==mode and r['withheld_family']==family and r['novelty_weight'] in [0,.95]],flush=True)
