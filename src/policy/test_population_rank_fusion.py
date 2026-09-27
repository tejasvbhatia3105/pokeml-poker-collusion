\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'population_rank_fusion';dest.mkdir(exist_ok=True)
cols=json.loads((root/'residual_columns.json').read_text());labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family')
allpairs=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='development').collect().sort('pair_id')
d=allpairs.join(labs,on='pair_id').sort('pair_id');X=d.select(cols).to_numpy();AX=allpairs.select(cols).to_numpy();y=d['label'].to_numpy();b=d['behavior_family'].to_numpy();g=d['table_id'].to_numpy();ag=allpairs['table_id'].to_numpy();folds=json.loads(Path('artifacts/folds.json').read_text());reports=[];outputs=[];t=time.time()
for family in ['all_known','directed_transfer','soft_play','coordinated_isolation']:
    pred={name:np.zeros(len(d)) for name in ['base','base_rarity','simple_rarity','monotone_rarity']}
    for f in folds:
        va=np.isin(g,f['valid_tables']);tr=~va & ((b!=family) if family!='all_known' else True);av=np.isin(ag,f['valid_tables']);ad=allpairs.filter(pl.Series(av))
        if family=='all_known':
            m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f["fold"]}.cbm'));base=1-m.predict_proba(AX[av])[:,0]
        else:
            m=CatBoostClassifier(iterations=600,depth=4,learning_rate=.045,loss_function='Logloss',l2_leaf_reg=10,verbose=False,thread_count=4,random_seed=318+f['fold'],allow_writing_files=False);m.fit(X[tr],y[tr]);base=m.predict_proba(AX[av])[:,1]
        param=np.load(root/f'monotone_novelty/scale_fold{f["fold"]}.npz');z=np.log1p(np.abs((ad.select(param['columns'].tolist()).to_numpy()-param['median'])/param['scale']));simple=np.sort(z,axis=1)[:,-3:].mean(1)
        m=CatBoostClassifier();m.load_model(str(root/f'monotone_novelty/{family}_monotone_fold{f["fold"]}.cbm'));monotone=m.predict_proba(z)[:,1]
        frame=pd.DataFrame(dict(pair_id=ad['pair_id'].to_list(),base=base))
        for name,score in [('base',base),('simple',simple),('monotone',monotone)]:
            frame[name+'_rarity']=-np.log((len(score)+1-rankdata(score,method='average'))/(len(score)+1))
        labeled=frame.set_index('pair_id').loc[d.filter(pl.Series(va))['pair_id'].to_list()]
        for name in pred:pred[name][va]=labeled[name].to_numpy()
        print(family,f['fold'],round(time.time()-t,1),flush=True)
    keep=np.ones(len(d),bool) if family=='all_known' else (y==0)|(b==family)
    for novel in ['simple','monotone']:
        for weight in [0,.5,.75,1]:
            score=np.maximum(pred['base_rarity'],weight*pred[novel+'_rarity'])
            reports.append(dict(withheld_family=family,novelty=novel,weight=weight,weighted_AP=ap(y[keep],score[keep],sample_weight=np.where(y[keep]>0,1,50))))
    outputs.append(pd.DataFrame(dict(pair_id=d['pair_id'].to_list(),withheld_family=family,truth=y,behavior=b,**pred)));pd.concat(outputs).to_csv(dest/'oof.csv',index=False);(dest/'metrics.json').write_text(json.dumps(reports,indent=2));print('RESULT',family,[r for r in reports if r['withheld_family']==family],flush=True)
