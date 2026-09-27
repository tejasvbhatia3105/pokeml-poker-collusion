\
\
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

C=pl.col; root=Path('artifacts/policy'); dest=root/'outcome_roles';dest.mkdir(exist_ok=True)
cols=json.loads((root/'feature_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text())
models=[]
for f in range(4):
    m=CatBoostClassifier();m.load_model(str(root/f'action_fold{f}.cbm'));models.append(m)

def build(table,query,chronological=False):
    a=pl.read_parquet(root/'actions'/f'{table}.parquet')
    needed=query.select('hand_id').unique();a=a.join(needed,on='hand_id',how='semi')
    if a.is_empty():return None
    dev=a['phase'].to_numpy()=='development';p=np.zeros((len(a),4));X=a.select(cols).to_numpy()
    for mask,fs in [(dev,[folds[table]]),(~dev,range(4))]:
        if mask.any():p[mask]=np.mean([models[f].predict_proba(X[mask],thread_count=4) for f in fs],axis=0)
    call=a['to_call'].to_numpy()>0;legal=np.ones_like(p);legal[call,1]=0;legal[~call,0]=0;legal[~call,2]=0
    act=a['action_class'].to_numpy();assert np.all(legal[np.arange(len(a)),act]>0)
    p*=legal;p/=p.sum(1,keepdims=True)
    a=a.with_columns(pl.Series('surprise',-np.log(p[np.arange(len(a)),act].clip(1e-7))),
        *[pl.Series(f'r{k}',(act==k)-p[:,k]) for k in range(4)],C('street_no').cast(pl.Int64))
    seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').select('hand_id','player_id','net_chips')
    query=query.join(seats.rename({'player_id':'player_1','net_chips':'net1'}),on=['hand_id','player_1']).join(seats.rename({'player_id':'player_2','net_chips':'net2'}),on=['hand_id','player_2'])
    mapping=pl.concat([query.select('pair_id','hand_id',C('player_1').alias('player_id'),C('player_2').alias('partner'),C('net1').alias('net'),C('net2').alias('other_net')),
        query.select('pair_id','hand_id',C('player_2').alias('player_id'),C('player_1').alias('partner'),C('net2').alias('net'),C('net1').alias('other_net'))])
    z=a.join(mapping,on=['hand_id','player_id']).sort('pair_id','hand_id','action_no')
    st=pl.read_parquet(root/'states'/f'{table}.parquet').select('hand_id','street_no',C('player_id').alias('partner'),C('fold_no').alias('partner_fold'),C('equity').alias('partner_eq'))
    z=z.join(st,on=['hand_id','street_no','partner']).with_columns((C('last_aggressor')==C('partner')).fill_null(False).alias('facing'),(C('partner_fold')>=C('action_no')).alias('alive'))
                                                                                
                                                                             
    if chronological:z=z.sort('pair_id','hand_id','action_no','player_id')
    z=z.with_columns((C('equity')-C('partner_eq')).alias('equity_gap'))
    expr=[]
    for role,rmask in [('lower',C('net')<=C('other_net')),('higher',C('net')>=C('other_net'))]:
        for street in range(4):
            mask=rmask&(C('street_no')==street);prefix=f'outcome_{role}_s{street}'
            for field in ['equity','partner_eq','equity_gap','made_category','made_kicker','hole_suit_matches','hole_board_matches']:
                expr.append(C(field).filter(mask).mean().fill_null(-1).alias(prefix+'_'+field))
            for k in range(4):
                for context,cmask in [('partner',C('facing')&C('alive')),('other',~C('facing')&C('alive'))]:
                    gate=mask&cmask; event=gate&(C('action_class')==k);name=prefix+f'_{context}_{k}'
                    expr.extend([event.sum().alias(name+'_count'),C(f'r{k}').filter(gate).sum().alias(name+'_residual'),
                        C('surprise').filter(event).max().fill_null(0).alias(name+'_surprise'),
                        C('log_amount_bb').filter(event).max().fill_null(0).alias(name+'_amount'),
                        C('equity_gap').filter(event).mean().fill_null(-2).alias(name+'_equity_gap')])
                                                                                              
            for field in ['action_class','pot_odds','call_stack','log_bet_ratio','players_active']:
                expr.append(C(field).filter(mask).last().fill_null(-1).alias(prefix+'_last_'+field))
    return z.group_by('pair_id','hand_id').agg(expr).with_columns(pl.selectors.numeric().cast(pl.Float32))

if __name__=='__main__':
    labs=pl.read_csv('data/development_labels.csv').filter(C('label')==1).select('pair_id','player_1','player_2')
    query=pl.read_parquet(root/'evidence_training.parquet').select('pair_id','hand_id','table_id').join(labs,on='pair_id')
    t=time.time();parts=[]
    for i,((table,),q) in enumerate(query.group_by('table_id')):
        out=dest/f'{table}.parquet'
        if not out.exists():
            d=build(table,q.select('pair_id','hand_id','player_1','player_2'));assert np.isfinite(d.select(pl.selectors.numeric()).to_numpy()).all();d.write_parquet(out,compression='zstd')
        if i%40==0:print('evidence features',i,round(time.time()-t,1),flush=True)
