import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');sys.path.insert(0,'src/policy')
from pathlib import Path
import json,time,argparse
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
from evidence_data import load
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['ordinal','multiway','censored']);args=ap.parse_args()
ROOT=Path('artifacts/evidence_session5');ROOT.mkdir(exist_ok=True);old=Path('artifacts/policy');C=pl.col
d,_=load();extra=pl.read_parquet(list((old/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((old/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
d=d.join(extra,on=['pair_id','hand_id']).join(pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
cols=json.loads((old/'relationship_evidence/columns.json').read_text())
if args.mode=='multiway':
    features=pl.read_parquet(ROOT/'multiway_features.parquet');d=d.join(features,on=['pair_id','hand_id'],validate='1:1');cols += [c for c in features.columns if c.startswith('multi_')]
elif args.mode=='ordinal':
    d=d.join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id','evidence_rank'),on=['pair_id','hand_id'],how='left').with_columns(C('evidence_rank').fill_null(0))
else:
    d=d.with_columns(C('evidence').sum().over('pair_id').alias('truth_count'),C('time').filter(C('evidence')==1).max().over('pair_id').alias('last_truth_time'))
(ROOT/f'{args.mode}_columns.json').write_text(json.dumps(cols));X=d.select(cols).to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();y=d['evidence'].to_numpy();parts=[];t=time.time()
for f in range(4):
    p=np.full(len(d),np.nan);graded=np.full(len(d),np.nan)
    for b in ['directed_transfer','soft_play','coordinated_isolation']:
        tr=(fv!=f)&(fam==b);va=(fv==f)&(fam==b)
        config=dict(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=1710+11*f,verbose=False,allow_writing_files=False)
        if args.mode=='censored':tr &= ((d['truth_count'].to_numpy()<5)|(d['time'].to_numpy()<=d['last_truth_time'].to_numpy()))
        target=d['evidence_rank'].to_numpy() if args.mode=='ordinal' else y
        m=CatBoostClassifier(**config,loss_function='MultiClass' if args.mode=='ordinal' else 'Logloss')
        m.fit(X[tr],target[tr]);pr=m.predict_proba(X[va]);p[va]=1-pr[:,0] if args.mode=='ordinal' else pr[:,1]
        if args.mode=='ordinal':graded[va]=pr@np.array([0 if c==0 else (6-c)/5 for c in m.classes_])
        m.save_model(str(ROOT/f'{args.mode}_{b}_fold{f}.cbm'))
    va=fv==f;q=d.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence','fold').with_columns(pl.Series('score',p[va]))
    if args.mode=='ordinal':q=q.with_columns(pl.Series('graded_score',graded[va]))
    parts.append(q);print(args.mode,'fold',f,'seconds',round(time.time()-t,1),flush=True)
out=pl.concat(parts)
if args.mode=='censored':
                                                                                  
                                                                               
                                                                               
    out=out.join(d.select('pair_id','hand_id','time'),on=['pair_id','hand_id']);rr=[]
    for _,g in out.group_by('pair_id'):
        g=g.sort('time','hand_id');state=np.array([1.,0.,0.,0.,0.]);scores=[]
        for p in g['score']:
            scores.append(p*state.sum());state=np.r_[state[0]*(1-p),state[1:]*(1-p)+state[:-1]*p]
        rr.append(g.with_columns(pl.Series('first5_score',scores)))
    out=pl.concat(rr)
out.write_parquet(ROOT/f'{args.mode}_oof.parquet')
