\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from catboost import CatBoostClassifier

root=Path('artifacts/policy');dest=root/'partial_partner';C=pl.col
core=json.loads((root/'residual_columns.json').read_text());folds=json.loads(Path('artifacts/folds.json').read_text())
labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family');cross=pd.read_csv(root/'crossed_validation/oof.csv');cross=cross[cross.training=='full_training'];outputs=[];reports=[]
full=pl.read_parquet(list((dest/'full').glob('*.parquet'))).filter(C('phase')=='development').join(labs,on='pair_id')
tables=pl.scan_parquet(str(root/'pair_features/*.parquet')).select('pair_id','table_id').unique().collect();full=full.join(tables,on='pair_id')
nc=[c for c in full.columns if c.endswith(('_t_min','_t_max'))]
for window in ['full','first_2000','last_2000']:
    folder=root/'pair_features' if window=='full' else root/'full_window_stress'/window/'pair_features'
    a=pl.scan_parquet(str(folder/'*.parquet')).filter(C('phase')=='development').collect().join(pl.read_parquet(list((dest/window).glob('*.parquet'))),on=['pair_id','phase']).sort('pair_id');parts=[]
    for f in folds:
        train=full.filter(~C('table_id').is_in(f['valid_tables'])&(C('label')==0)).select(nc).to_numpy();med=np.median(train,axis=0);scale=np.maximum(np.quantile(train,.9,axis=0)-np.quantile(train,.1,axis=0),.01)
        q=a.filter(C('table_id').is_in(f['valid_tables']));z=np.log1p(abs((q.select(nc).to_numpy()-med)/scale));score=z.max(1);rarity=-np.log1p(-rankdata(score)/(len(score)+1))
        m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));risk=1-m.predict_proba(q.select(core).to_numpy(),thread_count=4)[:,0];br=-np.log1p(-rankdata(risk)/(len(risk)+1))
        parts.append(pd.DataFrame(dict(pair_id=q['pair_id'].to_list(),conditional_score=score,conditional_rarity=rarity,v4_rarity=br)))
    d=cross[cross.window==window].merge(pd.concat(parts),on='pair_id',validate='many_to_one');outputs.append(d)
    for family,q in d.groupby('withheld_family'):
        if family!='all_known':q=q[(q.truth==0)|(q.behavior==family)]
        base=q.v4_rarity if family=='all_known' else q.rarity
        variants={'base':base,'v5':np.maximum(base,.95*q.novelty),'partial_floor':np.maximum(base,.95*q.conditional_rarity),'partial_alone':q.conditional_rarity}
        for name,score in variants.items():reports.append(dict(window=window,family=family,model=name,weighted_AP=ap(q.truth,score,sample_weight=np.where(q.truth>0,1,50))))
pd.concat(outputs).to_csv(dest/'novelty_oof.csv',index=False);pd.DataFrame(reports).to_csv(dest/'novelty_metrics.csv',index=False)
print(pd.DataFrame(reports).pivot(index=['window','family'],columns='model',values='weighted_AP').to_string(),flush=True)
