\
\
\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','8')
import json,time,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
C=pl.col; V1=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(exist_ok=True,parents=True); root=Path('artifacts/policy'); A=24
cols=json.loads((root/'feature_columns.json').read_text()); folds=json.loads((root/'table_folds.json').read_text())
models=[]
for f in range(4):
    m=CatBoostClassifier(); m.load_model(str(root/f'action_fold{f}.cbm')); models.append(m)
FEAT=['cls','street','lamt','lratio','lpot','lcall','lstack','pot_odds','call_stack','position','nact','prior_raises','prev','eq','made','p0','p1','p2','p3','rtaken','surp','self_la','tpos','tocall']
json.dump(FEAT,open(OUT/'action_columns.json','w')); t0=time.time()
for i,p in enumerate(sorted(V1.glob('*.npz'))):
    table=p.stem; out=OUT/f'{table}.npz'
    if out.exists(): continue
    z=np.load(p)
    a=pl.read_parquet(root/'actions'/f'{table}.parquet').sort('hand_id','action_no')
    players=sorted(set(a['player_id'].to_list())|set(z['player_1'].tolist())|set(z['player_2'].tolist())); pidx={q:k for k,q in enumerate(players)}
    X=a.select(cols).to_numpy(); dev=a['phase'].to_numpy()=='development'; act=a['action_class'].to_numpy(); pr=np.zeros((len(a),4))
    for mask,fs in [(dev,[folds[table]]),(~dev,range(4))]:
        if mask.any(): pr[mask]=np.mean([models[f].predict_proba(X[mask],thread_count=8) for f in fs],axis=0)
    call=a['to_call'].to_numpy()>0; legal=np.ones_like(pr); legal[call,1]=0; legal[~call,0]=0; legal[~call,2]=0; inv=legal[np.arange(len(a)),act]==0; legal[inv]=1; pr*=legal; pr/=pr.sum(1,keepdims=True); pt=pr[np.arange(len(a)),act]
    a=a.with_columns(*[pl.Series(f'p{k}',pr[:,k]) for k in range(4)],pl.Series('rtaken',1-pt),pl.Series('surp',-np.log(pt.clip(1e-7))),
        C('action_class').cast(pl.Float32).alias('cls'),C('street_no').cast(pl.Float32).alias('street'),C('log_amount_bb').alias('lamt'),C('log_bet_ratio').alias('lratio'),(C('pot_bb')+1).log().alias('lpot'),(C('call_bb')+1).log().alias('lcall'),(C('stack_bb')+1).log().alias('lstack'),
        C('players_active').cast(pl.Float32).alias('nact'),C('previous_action').cast(pl.Float32).alias('prev'),C('equity').alias('eq'),C('made_category').alias('made'),C('self_last_aggressor').alias('self_la'),(C('to_call')>0).cast(pl.Float32).alias('tocall'),
        (C('action_no')/C('action_no').max().over('hand_id').clip(1)).alias('tpos'),C('action_no').rank('ordinal').over('hand_id').alias('k'))
    a=a.filter(C('k')<=A)
    hands=a.select('hand_id','time_index').unique().sort('time_index'); hid2i={h:k for k,h in enumerate(hands['hand_id'])}; H=len(hands)
    XA=np.zeros((H,A,len(FEAT)),np.float16); ACT=np.full((H,A),-1,np.int16); LAG=np.full((H,A),-1,np.int16)
    hi=np.array([hid2i[h] for h in a['hand_id']]); ki=(a['k'].to_numpy()-1).astype(int)
    XA[hi,ki]=a.select(FEAT).to_numpy().astype(np.float16); ACT[hi,ki]=np.array([pidx[q] for q in a['player_id']],np.int16)
    la=a['last_aggressor'].to_list(); LAG[hi,ki]=np.array([pidx.get(q,-1) if q is not None else -1 for q in la],np.int16)
    hidx=np.array([hid2i.get(h,-1) for h in z['hand_id']],np.int32); assert (hidx>=0).all()
    p1=np.array([pidx[q] for q in z['player_1']],np.int16); p2=np.array([pidx[q] for q in z['player_2']],np.int16)
    np.savez(out,XA=XA,ACT=ACT,LAG=LAG,hidx=hidx,p1=p1,p2=p2)
    if i%40==0: print(i,table,H,round(time.time()-t0),flush=True)
print('done',round(time.time()-t0),flush=True)
