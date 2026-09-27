\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
import build_relationship_evidence as BRE
ROOT=Path('artifacts/evidence_session6');C=pl.col;CHANNELS=['call','fold','raise','outside_agg','outside_fold','hu_check']
def build(table,query):
    a=pl.read_parquet(BRE.root/'actions'/f'{table}.parquet').join(query.select('hand_id').unique(),on='hand_id',how='semi');dev=a['phase'].to_numpy()=='development';X=a.select(BRE.cols).to_numpy();p=np.zeros((len(a),4))
    for mask,fs in [(dev,[BRE.folds[table]]),(~dev,range(4))]:
        if mask.any():p[mask]=np.mean([BRE.models[f].predict_proba(X[mask],thread_count=4) for f in fs],axis=0)
    act=a['action_class'].to_numpy();call=a['to_call'].to_numpy()>0;legal=np.ones_like(p);legal[call,1]=0;legal[~call,0]=0;legal[~call,2]=0
    assert np.all(legal[np.arange(len(a)),act]>0);p*=legal;p/=p.sum(1,keepdims=True)
    a=a.with_columns(*[pl.Series(f'r{k}',(act==k)-p[:,k]) for k in range(4)],C('street_no').cast(pl.Int64))
    mp=pl.concat([query.select('pair_id','hand_id',C('player_1').alias('player_id'),C('player_2').alias('partner'),pl.lit(1).alias('sign')),query.select('pair_id','hand_id',C('player_2').alias('player_id'),C('player_1').alias('partner'),pl.lit(-1).alias('sign'))])
    z=a.join(mp,on=['hand_id','player_id']);st=pl.read_parquet(BRE.root/'states'/f'{table}.parquet').select('hand_id','street_no',C('player_id').alias('partner'),C('fold_no').alias('partner_fold'));z=z.join(st,on=['hand_id','street_no','partner'])
    alive=C('partner_fold')>=C('action_no');facing=(C('last_aggressor')==C('partner')).fill_null(False)&alive;outside=alive&~facing&(C('players_active')>=3);hu=alive&(C('players_active')==2)
    channels={'call':(2,facing),'fold':(0,facing),'raise':(3,facing),'outside_agg':(3,outside),'outside_fold':(0,outside),'hu_check':(1,hu)};expr=[]
    for name,(k,mask) in channels.items():
        for prefix,coef in [('sum',pl.lit(1)),('diff',C('sign'))]:expr.append(pl.when(mask).then(coef*C(f'r{k}')).otherwise(0).sum().alias(prefix+'_'+name))
    h=z.group_by('pair_id','hand_id','time_index').agg(expr).sort('pair_id','time_index','hand_id');parts=[]
    for (pid,),g in h.group_by('pair_id',maintain_order=True):
        total=g.select(['sum_'+n for n in CHANNELS]).to_numpy();diff=g.select(['diff_'+n for n in CHANNELS]).to_numpy();t=g['time_index'].to_numpy().astype(float);dt=t[:,None]-t[None,:];out={}
        for j,n in enumerate(CHANNELS):out['episode_current_shared_'+n]=total[:,j]
        for width in [25,100,400]:
            kernel=np.exp(-np.abs(dt)/width);np.fill_diagonal(kernel,0)
            for side,mask in [('past',dt>0),('future',dt<0),('both',np.ones_like(dt,bool))]:
                w=kernel*mask;rs=(w@total)/np.sqrt(1+(w*w)@(total*total));rd=(w@diff)/np.sqrt(1+(w*w)@(diff*diff));prefix=f'episode_{width}_{side}_';out[prefix+'exposure']=np.log1p(w.sum(1))
                values={'shared_reference':rs,'shared_coherence':total*rs,'directional_coherence':diff*rd,'cross_player_coherence':.5*(total*rs-diff*rd),'reference_direction_strength':np.abs(rd),'reference_shared_strength':np.abs(rs)}
                for name,v in values.items():
                    for j,n in enumerate(CHANNELS):out[prefix+name+'_'+n]=v[:,j]
        parts.append(g.select('pair_id','hand_id').hstack(pl.DataFrame(out).with_columns(pl.all().cast(pl.Float32))))
    return pl.concat(parts)
if __name__=='__main__':
    ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet');labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');parts=[];start=time.time()
    for i,((table,),q) in enumerate(ix.sort('table_id','pair_id','hand_id').group_by('table_id',maintain_order=True)):
        query=q.select('pair_id','hand_id').join(labs,on='pair_id');r=build(table,query)
        if i==0:
            swapped=build(table,query.rename({'player_1':'player_2','player_2':'player_1'})).sort('pair_id','hand_id');original=r.sort('pair_id','hand_id');assert np.allclose(original.select(pl.selectors.numeric()).to_numpy(),swapped.select(pl.selectors.numeric()).to_numpy(),atol=1e-6)
        parts.append(r)
        if i%60==0:print('episode features',i,'seconds',round(time.time()-start,1),flush=True)
    out=pl.concat(parts);assert len(out)==len(ix);out.write_parquet(ROOT/'episode_features.parquet');print(out.shape,flush=True)
