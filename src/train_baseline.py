import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import polars as pl,numpy as np,pandas as pd,json,time
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score,confusion_matrix
from catboost import CatBoostClassifier
EXCLUDE=['pair_id','hand_id','phase','table_id']
def ap(y,s):
    order=np.argsort(-s,kind='stable');r=np.asarray(y)[order];return float((np.cumsum(r)/np.arange(1,len(r)+1)*r).sum()/max(r.sum(),1))
def metric(y,p):
    risk=1-p[:,0];pred=p[:,1:].argmax(1)+1
    return {'pair_ap':ap(y>0,risk),'behavior_map':np.mean([ap(y==i,risk*(pred==i)) for i in range(1,4)])}
def aggregates(df):
    cols=[c for c in df.columns if c not in EXCLUDE]
    expr=[pl.len().alias('n_hands')]
    for c in cols:
        if c=='time':continue
        e=pl.col(c)
        expr += [e.mean().alias(c+'_mean'),e.max().alias(c+'_max'),e.std().alias(c+'_std')]
        if c not in ['net_direction','seat_distance','min_stack','max_stack']:
            expr+=[e.top_k(5).mean().alias(c+'_top5'),(e>0).mean().alias(c+'_rate')]
    return df.group_by('pair_id','phase','table_id').agg(expr).fill_null(0)
if __name__=='__main__':
    t=time.time();cache=Path('artifacts/pair_features.parquet')
    if not cache.exists():
        parts=[]
        for path in sorted(Path('artifacts/hand_features').glob('*.parquet')):
            parts.append(aggregates(pl.read_parquet(path)))
        pl.concat(parts).write_parquet(cache)
    f=pl.read_parquet(cache); labs=pl.read_csv('data/development_labels.csv')
    train=f.filter(pl.col('phase')=='development').join(labs,on='pair_id').sort('pair_id')
    names=['none','directed_transfer','soft_play','coordinated_isolation']; y=np.array([names.index(v) for v in train['behavior_family']]); groups=train['table_id'].to_numpy()
    cols=[c for c in f.columns if c not in ['pair_id','phase','table_id']];X=train.select(cols).to_numpy();oof=np.zeros((len(y),4));models=[]
    split=StratifiedGroupKFold(4,shuffle=True,random_state=42)
    folds=[]
    for i,(tr,va) in enumerate(split.split(X,y,groups)):
        m=CatBoostClassifier(iterations=650,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=6,verbose=False,thread_count=6,random_seed=42+i)
        m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=100)
        oof[va]=m.predict_proba(X[va]);print(i,m.best_iteration_,metric(y[va],oof[va]),flush=True)
        m.save_model(f'artifacts/pair_baseline_fold{i}.cbm');models.append(m)
        folds.append({'fold':i,'valid_tables':sorted(set(groups[va]))})
    print('OOF',metric(y,oof),'seconds',time.time()-t,flush=True)
    print(confusion_matrix(y,oof.argmax(1)),flush=True)
    Path('artifacts/folds.json').write_text(json.dumps(folds,indent=2))
    Path('artifacts/baseline_metrics.json').write_text(json.dumps(metric(y,oof),indent=2))
    pd.DataFrame({'feature':cols,'importance':np.mean([m.feature_importances_ for m in models],0)}).sort_values('importance',ascending=False).to_csv('artifacts/pair_importance.csv',index=False)
    pd.DataFrame(oof,columns=names).assign(pair_id=train['pair_id'].to_list(),truth=y).to_csv('artifacts/pair_oof.csv',index=False)
    eva=f.filter(pl.col('phase')=='evaluation').sort('pair_id')
    p=np.mean([m.predict_proba(eva.select(cols).to_numpy()) for m in models],axis=0)
    pd.DataFrame(p,columns=names).assign(pair_id=eva['pair_id'].to_list()).to_csv('artifacts/pair_baseline_eval.csv',index=False)
