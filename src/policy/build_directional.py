\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
from pathlib import Path
import json, time
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from window_actions import window_actions

C = pl.col
root = Path('artifacts/policy')
dest = root/'directional'
cols = json.loads((root/'feature_columns.json').read_text())
folds = json.loads((root/'table_folds.json').read_text())
models = []
for f in range(4):
    m = CatBoostClassifier(); m.load_model(str(root/f'action_fold{f}.cbm')); models.append(m)
labels = pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
evaluation = pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2')
known = pl.concat([labels,evaluation])
t = time.time()

def build(a, st, pairs):
    X = a.select(cols).to_numpy(); act = a['action_class'].to_numpy()
    dev = a['phase'].to_numpy() == 'development'; p = np.zeros((len(a),4))
    for mask, fs in [(dev,[folds[table]]),(~dev,range(4))]:
        if mask.any(): p[mask] = np.mean([models[f].predict_proba(X[mask],thread_count=4) for f in fs],axis=0)
    call = a['to_call'].to_numpy()>0
    legal=np.ones_like(p); legal[call,1]=0; legal[~call,0]=0; legal[~call,2]=0
    assert np.all(legal[np.arange(len(a)),act]>0)
    p*=legal; p/=p.sum(1,keepdims=True)
    r=np.eye(4)[act]-p
    a=a.with_columns(*[pl.Series(f'r{k}',r[:,k]) for k in range(4)],
        *[pl.Series(f'v{k}',p[:,k]*(1-p[:,k])) for k in range(4)],C('street_no').cast(pl.Int64))
    a=a.with_columns(*[(C(f'r{k}')-C(f'r{k}').mean().over(['player_id','phase','street_no'])).alias(f'r{k}') for k in range(4)])
    z=a.join(st,on=['hand_id','street_no']).filter(C('player_id')!=C('partner'))
    z=z.with_columns(pl.min_horizontal('player_id','partner').alias('player_1'),pl.max_horizontal('player_id','partner').alias('player_2'))
    z=z.join(pairs,on=['player_1','player_2']).with_columns((C('player_id')==C('player_1')).cast(pl.Int8).alias('role'))
    alive=C('partner_fold')>=C('action_no'); facing=(C('last_aggressor')==C('partner')).fill_null(False)&alive
    hu=alive&(C('players_active')==2); out=alive&~facing&(C('players_active')>=3)
    channels={
        'facing_call':(2,facing,pl.lit(1)),
        'facing_fold':(0,facing,pl.lit(1)),
        'facing_raise':(3,facing,pl.lit(1)),
        'weak_call':(2,facing&(C('equity')<.45),pl.lit(1)),
        'strong_fold':(0,facing&(C('equity')>.65),pl.lit(1)),
        'hu_agg':(3,hu,pl.lit(1)),
        'hu_check':(1,hu,pl.lit(1)),
        'outside_agg':(3,out,pl.lit(1)),
        'outside_fold':(0,out,pl.lit(1)),
        'hidden_agg':(3,alive,C('partner_eq')-.5),
        'hidden_fold':(0,alive,C('partner_eq')-.5),
        'hidden_folded_agg':(3,~alive,C('partner_eq')-.5),
    }
    expr=[]; names=[]
    for street in [-1,0,1,2,3]:
        for name,(k,mask,coef) in channels.items():
            n=f'dir_s{street+1}_{name}'; names.append(n)
            mask=mask & (pl.lit(True) if street==-1 else C('street_no')==street)
            expr.extend([pl.when(mask).then(C(f'r{k}')*coef).otherwise(0).sum().alias(n+'_r'),
                         pl.when(mask).then((C(f'v{k}')+.01)*coef*coef).otherwise(0).sum().alias(n+'_v')])
    q=z.group_by('pair_id','phase','role').agg(expr)
    q=q.with_columns(*[(C(n+'_r')/(C(n+'_v')+1).sqrt()).alias(n) for n in names])
                                                                                               
    keys=['pair_id','phase']; base=z.select(keys).unique()
    first=q.filter(C('role')==0).select(*keys,*[C(n).alias(n+'_a') for n in names])
    second=q.filter(C('role')==1).select(*keys,*[C(n).alias(n+'_b') for n in names])
    q=base.join(first,on=keys,how='left').join(second,on=keys,how='left').fill_null(0)
    expr=[]
    for n in names:
        aa=C(n+'_a'); bb=C(n+'_b')
        expr.extend([pl.min_horizontal(aa,bb).alias(n+'_min'),pl.max_horizontal(aa,bb).alias(n+'_max'),
                     (aa-bb).abs().alias(n+'_asym'),(aa*bb).alias(n+'_reciprocal')])
                                                                                             
    for left,right in [('outside_agg','facing_fold'),('facing_raise','facing_call'),('hu_agg','hu_check'),('outside_agg','hidden_fold')]:
        for early,late in [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]:
            x=f'dir_s{early}_{left}'; y=f'dir_s{late}_{right}'
            expr.append(((C(x+'_a')-C(x+'_b'))*(C(y+'_a')-C(y+'_b'))).alias(f'dir_transition_{early}_{late}_{left}_{right}'))
    return q.select(*keys,*expr).with_columns(pl.selectors.float().cast(pl.Float32)).fill_null(0)

for i,path in enumerate(sorted((root/'actions').glob('*.parquet'))):
    table=path.stem
    st=pl.read_parquet(root/'states'/path.name).select('hand_id','street_no',C('player_id').alias('partner'),C('fold_no').alias('partner_fold'),C('equity').alias('partner_eq'))
    original=pl.read_parquet(path)
    pool=pl.read_parquet(root/'pair_features'/path.name).select('pair_id').unique()
    for window in ['full','first_2000','last_2000']:
        folder=dest/window; folder.mkdir(exist_ok=True,parents=True); out=folder/path.name
        if out.exists():continue
        a=original if window=='full' else window_actions(original,window)
        pairs=(known if window=='full' else labels).join(pool,on='pair_id',how='semi')
        if pairs.is_empty():continue
        data=build(a,st,pairs)
        assert np.isfinite(data.select(pl.selectors.numeric()).to_numpy()).all()
        data.write_parquet(out,compression='zstd')
    if i%10==0:print('built',i,table,round(time.time()-t,1),flush=True)
    if os.environ.get('ONE_TABLE'):break
