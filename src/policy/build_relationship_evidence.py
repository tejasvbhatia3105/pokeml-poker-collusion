\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time
import numpy as np
import polars as pl
from catboost import CatBoostClassifier

root=Path('artifacts/policy'); dest=root/'relationship_evidence';dest.mkdir(exist_ok=True); C=pl.col
cols=json.loads((root/'feature_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text())
models=[]
for f in range(4):
    m=CatBoostClassifier();m.load_model(str(root/f'action_fold{f}.cbm'));models.append(m)

def build(table,query,chronological=False):
    a=pl.read_parquet(root/'actions'/f'{table}.parquet').join(query.select('hand_id').unique(),on='hand_id',how='semi')
    dev=a['phase'].to_numpy()=='development';X=a.select(cols).to_numpy();p=np.zeros((len(a),4))
    for mask,fs in [(dev,[folds[table]]),(~dev,range(4))]:
        if mask.any():p[mask]=np.mean([models[f].predict_proba(X[mask],thread_count=4) for f in fs],axis=0)
    act=a['action_class'].to_numpy();call=a['to_call'].to_numpy()>0;legal=np.ones_like(p);legal[call,1]=0;legal[~call,0]=0;legal[~call,2]=0
    assert np.all(legal[np.arange(len(a)),act]>0);p*=legal;p/=p.sum(1,keepdims=True)
    a=a.with_columns(*[pl.Series(f'r{k}',(act==k)-p[:,k]) for k in range(4)],C('street_no').cast(pl.Int64))
    mp=pl.concat([query.select('pair_id','hand_id',C('player_1').alias('player_id'),C('player_2').alias('partner'),pl.lit(1).alias('sign')),
        query.select('pair_id','hand_id',C('player_2').alias('player_id'),C('player_1').alias('partner'),pl.lit(-1).alias('sign'))])
    z=a.join(mp,on=['hand_id','player_id']);st=pl.read_parquet(root/'states'/f'{table}.parquet').select('hand_id','street_no',C('player_id').alias('partner'),C('fold_no').alias('partner_fold'))
    z=z.join(st,on=['hand_id','street_no','partner']);alive=C('partner_fold')>=C('action_no');facing=(C('last_aggressor')==C('partner')).fill_null(False)&alive
    outside=alive&~facing&(C('players_active')>=3);hu=alive&(C('players_active')==2)
    channels={'call':(2,facing),'fold':(0,facing),'raise':(3,facing),'outside_agg':(3,outside),'outside_fold':(0,outside),'hu_check':(1,hu)}
    expr=[]
    for name,(k,mask) in channels.items():
        expr.append(pl.when(mask).then(C('sign')*C(f'r{k}')).otherwise(0).sum().alias(name))
    h=z.group_by('pair_id','hand_id','time_index').agg(expr).sort('pair_id','time_index')
    seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').select('hand_id','player_id','net_chips')
    q=query.join(seats.rename({'player_id':'player_1','net_chips':'net1'}),on=['hand_id','player_1']).join(seats.rename({'player_id':'player_2','net_chips':'net2'}),on=['hand_id','player_2'])
    q=q.select('pair_id','hand_id',(C('net1')-C('net2')).sign().alias('net_direction'))
    h=h.join(q,on=['pair_id','hand_id']);expr=[]
                                                                              
                                                                                
    if chronological:h=h.sort('pair_id','time_index','hand_id')
    for name in channels:
        x=C(name);other=x.sum().over('pair_id')-x
        scale=(x.pow(2).sum().over('pair_id')-x.pow(2)+1).sqrt()
        reference=other/scale
        near=(x.rolling_sum(11,min_samples=1,center=True).over('pair_id')-x)/((x.pow(2).rolling_sum(11,min_samples=1,center=True).over('pair_id')-x.pow(2)+1).sqrt())
        expr.extend([(x*reference).alias('relationship_'+name+'_coherence'),
                     (C('net_direction')*reference).alias('relationship_'+name+'_payoff_alignment'),
                     (x*near).alias('relationship_'+name+'_local_coherence'),
                     (C('net_direction')*near).alias('relationship_'+name+'_local_payoff'),
                     reference.abs().alias('relationship_'+name+'_reference_strength')])
    return h.select('pair_id','hand_id',*expr).with_columns(pl.selectors.numeric().cast(pl.Float32))

if __name__=='__main__':
    labs=pl.read_csv('data/development_labels.csv').filter(C('label')==1).select('pair_id','player_1','player_2')
    q=pl.read_parquet(root/'evidence_training.parquet').select('pair_id','hand_id','table_id').join(labs,on='pair_id');t=time.time()
    for i,((table,),query) in enumerate(q.group_by('table_id')):
        out=dest/f'{table}.parquet'
        if not out.exists():
            d=build(table,query.select('pair_id','hand_id','player_1','player_2'));assert np.isfinite(d.select(pl.selectors.numeric()).to_numpy()).all();d.write_parquet(out,compression='zstd')
        if i%50==0:print('relationship evidence',i,round(time.time()-t,1),flush=True)
