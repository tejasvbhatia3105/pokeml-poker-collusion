\
\
\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','8')
import json,time,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
C=pl.col; V1=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(exist_ok=True,parents=True); root=Path('artifacts/policy')
cols=json.loads((root/'feature_columns.json').read_text()); folds=json.loads((root/'table_folds.json').read_text())
models=[]
for f in range(4):
    m=CatBoostClassifier(); m.load_model(str(root/f'action_fold{f}.cbm')); models.append(m)
FIELDS=['cls','agg','rtaken','surp','facing','lpot','lcall','nact','eq']; SEC=['cls','surp','facing','rtaken']
names2=[f'p{k}_s{s}_d0_{f}' for k in (1,2) for s in range(4) for f in FIELDS]+[f'p{k}_s{s}_d1_{f}' for k in (1,2) for s in range(4) for f in SEC]
json.dump(names2,open(OUT/'columns2.json','w')); t0=time.time()
for i,p in enumerate(sorted(V1.glob('*.npz'))):
    table=p.stem; out=OUT/f'{table}.npy'
    if out.exists(): continue
    z=np.load(p); off=np.concatenate([[0],np.cumsum(z['n'])])
    rows=pl.DataFrame({'row':np.arange(len(z['hand_id'])),'hand_id':z['hand_id'],'player_1':np.repeat(z['player_1'],z['n']),'player_2':np.repeat(z['player_2'],z['n'])})
    a=pl.read_parquet(root/'actions'/f'{table}.parquet')
    X=a.select(cols).to_numpy(); dev=a['phase'].to_numpy()=='development'; act=a['action_class'].to_numpy(); pr=np.zeros((len(a),4))
    for mask,fs in [(dev,[folds[table]]),(~dev,range(4))]:
        if mask.any(): pr[mask]=np.mean([models[f].predict_proba(X[mask],thread_count=8) for f in fs],axis=0)
    call=a['to_call'].to_numpy()>0; legal=np.ones_like(pr); legal[call,1]=0; legal[~call,0]=0; legal[~call,2]=0; inv=legal[np.arange(len(a)),act]==0; legal[inv]=1; pr*=legal; pr/=pr.sum(1,keepdims=True)
    ptaken=pr[np.arange(len(a)),act]
    a=a.with_columns(pl.Series('rtaken',1-ptaken),pl.Series('surp',-np.log(ptaken.clip(1e-7))),(C('action_class')==3).cast(pl.Float32).alias('agg'),C('action_class').cast(pl.Float32).alias('cls'),(C('pot_bb')+1).log().alias('lpot'),(C('call_bb')+1).log().alias('lcall'),C('players_active').cast(pl.Float32).alias('nact'),C('equity').cast(pl.Float32).alias('eq'),C('street_no').cast(pl.Int64))
    a=a.with_columns((C('action_no').rank('ordinal').over(['hand_id','player_id','street_no'])-1).alias('dec')).filter(C('dec')<2)
    wide=a.select('hand_id','player_id','street_no','dec','cls','agg','rtaken','surp','lpot','lcall','nact','eq','last_aggressor')
    parts=[]
    for k,pk in ((1,'player_1'),(2,'player_2')):
        other='player_2' if k==1 else 'player_1'
        j=rows.join(wide.rename({'player_id':pk}),on=['hand_id',pk],how='inner').with_columns((C('last_aggressor')==C(other)).fill_null(False).cast(pl.Float32).alias('facing'))
        ex=[]
        for s in range(4):
            for d,fl in ((0,FIELDS),(1,SEC)):
                m=(C('street_no')==s)&(C('dec')==d)
                for f in fl: ex.append(C(f).filter(m).first().alias(f'p{k}_s{s}_d{d}_{f}'))
        parts.append(j.group_by('row').agg(ex))
    W=rows.select('row').join(parts[0],on='row',how='left').join(parts[1],on='row',how='left').sort('row').select(names2).fill_null(-1.0)
    X2=W.to_numpy().astype(np.float16); assert X2.shape[0]==len(z['hand_id']); np.save(out,X2)
    if i%40==0: print(i,table,X2.shape,round(time.time()-t0),flush=True)
print('done',round(time.time()-t0),flush=True)
