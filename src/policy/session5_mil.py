\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session5_multiway_features import build
ROOT=Path('artifacts/evidence_session5');C=pl.col
idx=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').with_row_index('bag_id')
if not (ROOT/'mil_actions.parquet').exists():
    labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');parts=[];start=time.time()
    for i,((table,),q) in enumerate(idx.group_by('table_id')):
        a=build(table,q.select('pair_id','hand_id').join(labs,on='pair_id'),return_actions=True)
                                                                                      
        cols=[c for c,dtype in a.schema.items() if dtype.is_numeric() or dtype==pl.Boolean]
        cols=[c for c in cols if c not in ['time_index','time_bin','amount','to_call','big_blind']]
        parts.append(a.select('pair_id','hand_id',*cols).join(q.select('pair_id','hand_id','bag_id','fold','evidence','behavior_family','time'),on=['pair_id','hand_id']))
        if i%80==0:print('actions',i,'seconds',round(time.time()-start,1),flush=True)
    pl.concat(parts).write_parquet(ROOT/'mil_actions.parquet')
d=pl.read_parquet(ROOT/'mil_actions.parquet');cols=[c for c in d.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family']]
X=d.select(cols).to_numpy().astype('float32');y=d['evidence'].to_numpy();g=d['bag_id'].to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();start=time.time();allp={0:[],2:[]}
for f in range(4):
    preds={0:np.zeros(len(d)),2:np.zeros(len(d))}
    for family in ['directed_transfer','soft_play','coordinated_isolation']:
        tr=np.flatnonzero((fv!=f)&(fam==family));va=np.flatnonzero((fv==f)&(fam==family));yt=y[tr];gt=g[tr]
        counts=np.bincount(gt,minlength=len(idx));bw=1/counts[gt];resp=yt.astype(float)
        for em in range(3):
                                                                               
            positive=np.flatnonzero(yt==1)
            xx=np.concatenate([X[tr],X[tr][positive]],0)
            yy=np.r_[np.zeros(len(tr)),np.ones(len(positive))]
            ww=np.r_[bw*(1-resp),bw[positive]*resp[positive]*5]
            keep=ww>1e-8
            m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=5310+11*f,verbose=False,allow_writing_files=False)
            m.fit(xx[keep],yy[keep],sample_weight=ww[keep])
            p=m.predict_proba(X[tr],thread_count=4)[:,1].clip(1e-7,1-1e-7)
            totals=np.bincount(gt,weights=p,minlength=len(idx));resp=np.where(yt==1,p/totals[gt],0.)
            if em in [0,2]:
                preds[em][va]=m.predict_proba(X[va],thread_count=4)[:,1]
                m.save_model(str(ROOT/f'mil_em{em}_{family}_fold{f}.cbm'))
    for em in [0,2]:
        q=d.filter(C('fold')==f).select('pair_id','hand_id').with_columns(pl.Series('p',preds[em][fv==f]))
        allp[em].append(q.group_by('pair_id','hand_id').agg(C('p').max().alias('score'),(1-(1-C('p')).product()).alias('noisyor_score')))
    print('MIL fold',f,'seconds',round(time.time()-start,1),flush=True)
for em,name in [(0,'action'),(2,'mil')]:pl.concat(allp[em]).write_parquet(ROOT/f'{name}_oof.parquet')
(ROOT/'mil_columns.json').write_text(json.dumps(cols,indent=2))
