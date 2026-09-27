\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from window_actions import window_actions

root=Path('artifacts/policy');dest=root/'partial_partner';C=pl.col
cols=json.loads((root/'feature_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text())
channels=[f'{s}_{k}' for s in ['pre','post'] for k in range(4)]
models=[]
for f in range(4):
    m=CatBoostClassifier();m.load_model(str(root/f'action_fold{f}.cbm'));models.append(m)

def regress(x,y,ridge=10.):
    x=x-x.mean(0);y=y-y.mean(0)
    cov=x.T@x; inv=np.linalg.inv(cov+ridge*np.eye(x.shape[1]))
    xy=x.T@y;beta=inv@xy
    residual=y-x@beta;influence=x@inv
    se=np.sqrt((influence**2).T@(residual**2)+1e-6)
    uni=xy/(np.diag(cov)[:,None]+ridge)
    return beta,beta/se,uni

def build(a,roster,pairs,table):
    X=a.select(cols).to_numpy();act=a['action_class'].to_numpy();dev=a['phase'].to_numpy()=='development';p=np.zeros((len(a),4))
    for mask,fs in [(dev,[folds[table]]),(~dev,range(4))]:
        if mask.any():p[mask]=np.mean([models[f].predict_proba(X[mask],thread_count=4) for f in fs],axis=0)
    call=a['to_call'].to_numpy()>0;legal=np.ones_like(p);legal[call,1]=0;legal[~call,0]=0;legal[~call,2]=0
    assert np.all(legal[np.arange(len(a)),act]>0);p*=legal;p/=p.sum(1,keepdims=True)
    a=a.with_columns(*[pl.Series(f'r{k}',(act==k)-p[:,k]) for k in range(4)],*[pl.Series(f'v{k}',p[:,k]*(1-p[:,k])) for k in range(4)])
    a=a.with_columns(*[(C(f'r{k}')-C(f'r{k}').mean().over(['player_id','phase','street_no'])).alias(f'r{k}') for k in range(4)])
    expr=[]
    for street,mask in [('pre',C('street_no')==0),('post',C('street_no')>0)]:
        for k in range(4):expr.append((C(f'r{k}').filter(mask).sum()/(C(f'v{k}').filter(mask).sum()+1).sqrt()).alias(f'{street}_{k}'))
    h=a.group_by('hand_id','player_id').agg(expr)
    roster=roster.join(a.select('hand_id','phase','time_index').unique(),on='hand_id').join(h,on=['hand_id','player_id'],how='left').fill_null(0)
    players=sorted(roster['player_id'].unique().to_list());pi={p:i for i,p in enumerate(players)};N=len(players);output=[]
    for phase in ['development','evaluation']:
        q=roster.filter(C('phase')==phase)
        if not len(q):continue
        hands=q.select('hand_id','time_index').unique().sort('time_index');hi={h:i for i,h in enumerate(hands['hand_id'])}
        present=np.zeros((len(hands),N));Y=np.zeros((len(hands),N,len(channels)))
        for row in q.select('hand_id','player_id',*channels).iter_rows():
            i=hi[row[0]];j=pi[row[1]];present[i,j]=1;Y[i,j]=row[2:]
        betas=np.zeros((N,N,len(channels)));ts=betas.copy();uni=betas.copy();local=betas.copy()
        times=hands['time_index'].to_numpy();blocks=(times-times.min())//500
        for actor in range(N):
            mask=present[:,actor]>0;others=np.arange(N)!=actor;x=present[mask][:,others];y=Y[mask,actor]
            if len(x)<20:continue
            b,t,u=regress(x,y);betas[actor,others]=b;ts[actor,others]=t;uni[actor,others]=u
            for block in np.unique(blocks):
                take=mask&(blocks==block)
                if take.sum()<35:continue
                _,bt,_=regress(present[take][:,others],Y[take,actor]);local[actor,others]=np.maximum(local[actor,others],abs(bt))
        for pid,p1,p2 in pairs.select('pair_id','player_1','player_2').iter_rows():
            i=pi[p1];j=pi[p2];record=dict(pair_id=pid,phase=phase)
            for k,c in enumerate(channels):
                b1,b2=betas[i,j,k],betas[j,i,k];t1,t2=ts[i,j,k],ts[j,i,k];u1,u2=uni[i,j,k],uni[j,i,k]
                values=dict(coef_min=min(b1,b2),coef_max=max(b1,b2),coef_asym=abs(b1-b2),t_min=min(t1,t2),t_max=max(t1,t2),
                    uni_min=min(u1,u2),uni_max=max(u1,u2),suppression_min=min(b1-u1,b2-u2),suppression_max=max(b1-u1,b2-u2),
                    local_max=max(local[i,j,k],local[j,i,k]))
                record.update({f'partial_{c}_{key}':value for key,value in values.items()})
            output.append(record)
    return pl.DataFrame(output).with_columns(pl.selectors.float().cast(pl.Float32))

if __name__=='__main__':
    t=time.time()
    with threadpool_limits(limits=1):
        for i,path in enumerate(sorted((root/'actions').glob('*.parquet'))):
            a=pl.read_parquet(path);roster=pl.read_parquet(root/'states'/path.name).select('hand_id','player_id').unique();pairs=pl.read_parquet(root/'pair_features'/path.name).select('pair_id','player_1','player_2').unique()
            for window in ['full','first_2000','last_2000']:
                folder=dest/window;folder.mkdir(exist_ok=True,parents=True);out=folder/path.name
                if out.exists():continue
                z=build(a if window=='full' else window_actions(a,window),roster,pairs,path.stem)
                assert np.isfinite(z.select(pl.selectors.numeric()).to_numpy()).all();z.write_parquet(out,compression='zstd')
            if i%20==0:print('partial partner',i,round(time.time()-t,1),flush=True)
            if os.environ.get('ONE_TABLE'):break
