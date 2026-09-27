import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'full_window_stress';cols=json.loads((root/'residual_columns.json').read_text());nc=[c for c in cols if c.endswith(('_z','_w10_max','_w10_min'))];folds=json.loads(Path('artifacts/folds.json').read_text());labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');C=pl.col
models=[]
for f in range(4):
    m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f}.cbm'));models.append(m)
reports=[];outputs=[]
for window in ['first_2000','last_2000']:
    paths=list((dest/window/'pair_features').glob('*.parquet'));assert len(paths)==400,(window,len(paths))
    a=pl.read_parquet(paths).sort('pair_id');d=a.join(labs,on='pair_id').join(pl.read_parquet(root/f'exposure_{window}_eligibility.parquet'),on='pair_id').filter(C('eligible')).sort('pair_id');X=a.select(cols).to_numpy();NX=a.select(nc).to_numpy();y=d['label'].to_numpy();g=d['table_id'].to_numpy();ag=a['table_id'].to_numpy()
    pred={name:np.zeros(len(d)) for name in ['v4','v5_frozen','matched_reference']}
    for f in folds:
        va=np.isin(g,f['valid_tables']);av=np.isin(ag,f['valid_tables']);tr=~va&(y==0);risk=1-models[f['fold']].predict_proba(X[av],thread_count=4)[:,0];q=rankdata(risk)/(len(risk)+1);rarity=-np.log1p(-q)
        frame=pd.DataFrame(dict(pair_id=a.filter(pl.Series(av))['pair_id'].to_list(),v4=risk))
        p=np.load(root/f'monotone_novelty/scale_fold{f["fold"]}.npz')
        assert p['columns'].tolist()==nc
        normals=d.filter(pl.Series(tr)).select(nc).to_numpy();med=np.median(normals,axis=0);scale=np.maximum(np.quantile(normals,.9,axis=0)-np.quantile(normals,.1,axis=0),.01)
        for name,center,spread in [('v5_frozen',p['median'],p['scale']),('matched_reference',med,scale)]:
            z=np.log1p(abs((NX[av]-center)/spread));novel=np.sort(z,axis=1)[:,-3:].mean(1);nr=-np.log1p(-rankdata(novel)/(len(novel)+1));frame[name]=np.maximum(rarity,.95*nr)
        frame=frame.set_index('pair_id').loc[d.filter(pl.Series(va))['pair_id'].to_list()]
        for name in pred:pred[name][va]=frame[name]
    for name,score in pred.items():
        reports.append(dict(window=window,model=name,AP=ap(y,score),weighted_AP=ap(y,score,sample_weight=np.where(y>0,1,50))))
    outputs.append(pd.DataFrame(dict(pair_id=d['pair_id'].to_list(),window=window,truth=y,behavior=d['behavior_family'].to_list(),**pred)))
    print(window,[r for r in reports if r['window']==window],flush=True)
pd.concat(outputs).to_csv(dest/'oof.csv',index=False);(dest/'metrics.json').write_text(json.dumps(reports,indent=2))
