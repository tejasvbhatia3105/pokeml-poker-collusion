import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import polars as pl,numpy as np,pandas as pd,json,time
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
from train_baseline import ap
NAMES=['none','directed_transfer','soft_play','coordinated_isolation']

def evidence_map(df,prob,truth):
    out=df.select('pair_id','hand_id').to_pandas(); scores=[]
    for k,n in enumerate(NAMES[1:],1):out[n]=prob[:,k]
    for pair,g in out.groupby('pair_id'):
        if pair not in truth:continue
        n,ev=truth[pair]; top=g.sort_values(n,ascending=False).hand_id.head(5).tolist();hit=0;score=0
        for i,h in enumerate(top,1):
            if h in ev:hit+=1;score+=hit/i
        scores.append((pair,n,score/min(5,len(ev)),top))
    return scores

if __name__=='__main__':
    t=time.time();cache=Path('artifacts/dev_hands.parquet')
    if not cache.exists():
        pl.scan_parquet('artifacts/hand_features/*.parquet').filter(pl.col('phase')=='development').collect().write_parquet(cache)
    d=pl.read_parquet(cache);labs=pl.read_csv('data/development_labels.csv');e=pl.read_csv('data/development_evidence.csv')
    d=d.join(labs.select('pair_id','behavior_family','label'),on='pair_id').join(e.select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(pl.col('evidence').fill_null(0)).sort('pair_id','hand_id')
    truth={}
    for pair,h,n in e.select('pair_id','hand_id','behavior_family').iter_rows():
        if pair not in truth:truth[pair]=(n,set())
        truth[pair][1].add(h)
    cols=[c for c in d.columns if c not in ['pair_id','hand_id','phase','table_id','behavior_family','label','evidence','time','net_direction']]
    X=d.select(cols).to_numpy();y=np.array([NAMES.index(n) if ev else 0 for n,ev in d.select('behavior_family','evidence').iter_rows()]);groups=d['table_id'].to_numpy()
    pair=d.select('pair_id','table_id','behavior_family').unique().sort('pair_id');yp=np.array([NAMES.index(n) for n in pair['behavior_family']]);gp=pair['table_id'].to_numpy()
                                                                                                             
    oof=np.zeros((len(d),4));records=[];models=[];folds=[]
    for i,(trp,vap) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=42).split(yp,yp,gp)):
        vt=set(gp[vap]);va=np.isin(groups,list(vt));tr=~va
        w=np.where((d['label'].to_numpy()==1)&(y==0),.15,1.)
        w=np.where(y>0,8.,w)
        m=CatBoostClassifier(iterations=550,depth=6,learning_rate=.045,loss_function='MultiClass',l2_leaf_reg=8,verbose=False,thread_count=6,random_seed=61+i)
        m.fit(X[tr],y[tr],sample_weight=w[tr],eval_set=(X[va],y[va]),early_stopping_rounds=65)
        oof[va]=m.predict_proba(X[va]);rr=evidence_map(d.filter(pl.Series(va)),oof[va],truth);records+=rr
        print('fold',i,'iter',m.best_iteration_,'map',np.mean([r[2] for r in rr]),'by_family',{n:np.mean([r[2] for r in rr if r[1]==n]) for n in NAMES[1:]},flush=True)
        m.save_model(f'artifacts/evidence_fold{i}.cbm');models.append(m);folds.append({'fold':i,'valid_tables':sorted(vt)})
    print('OOF evidence',np.mean([r[2] for r in records]),'seconds',time.time()-t,flush=True)
    Path('artifacts/evidence_folds.json').write_text(json.dumps(folds,indent=2));Path('artifacts/hand_feature_columns.json').write_text(json.dumps(cols))
    d.select('pair_id','hand_id').with_columns([pl.Series(n,oof[:,i]) for i,n in enumerate(NAMES)]).write_parquet('artifacts/evidence_oof.parquet')
    pd.DataFrame(records,columns=['pair_id','behavior','map5','hands']).to_csv('artifacts/evidence_validation.csv',index=False)
    pd.DataFrame({'feature':cols,'importance':np.mean([m.feature_importances_ for m in models],0)}).sort_values('importance',ascending=False).to_csv('artifacts/hand_importance.csv',index=False)
    Path('artifacts/evidence_metrics.json').write_text(json.dumps({'evidence_map5':np.mean([r[2] for r in records]),'by_family':{n:np.mean([r[2] for r in records if r[1]==n]) for n in NAMES[1:]}},indent=2))
