\
\
\
\
\
\
\
\
import json,time
from functools import lru_cache
from pathlib import Path
import numpy as np
import polars as pl
import session111_native_rollout as n
import session112_integrated_values as v

ROOT=Path('artifacts/evidence_session114_actor_coupling')
@lru_cache(maxsize=2)
def uniforms(reps):return np.random.default_rng(11401).random((reps,4,6,128,2))

def simulate(roots,action,sizer,meta,reps=256):
    count=len(roots)*2*reps;states=np.repeat(np.stack([n.state(r['state']) for r in roots]),2*reps,axis=0)
    ri=np.repeat(np.arange(len(roots)),2*reps);sc=np.tile(np.repeat(np.arange(2),reps),len(roots));rep=np.tile(np.arange(reps),2*len(roots))
    templates=np.stack([r['templates'][:reps] for r in roots]);ranks=np.ascontiguousarray(np.stack([r['ranks'][:reps] for r in roots])[ri,rep],dtype=np.uint32)
    u=uniforms(reps);centers=np.array(meta['centers']);visits=np.zeros((count,4,6),np.int32)
    actors=np.empty(count,np.int32);dynamic=np.zeros((count,13),np.float32);legal=np.zeros((count,4),np.uint8);params=np.zeros((count,4))
    forced_k=np.array([r['forced_class'] for r in roots],np.int32)[ri];forced_amount=np.array([r['forced_amount'] for r in roots],np.float64)[ri];total=0;maxstep=0
    for step in range(n.f.s.CONFIG['max_actions']):
        n.LIB.decisions(states,count,actors);active=actors>=0
        if not active.any():break
        maxstep=step+1;n.LIB.dynamic_features(states,count,actors,dynamic,legal,params)
        ids=np.flatnonzero(active);streets=states[:,43].astype(int)
        kinds=np.where(params[:,0]>0,2,1).astype(np.int32);amounts=params[:,0].copy()
        nn=np.flatnonzero(active&(sc==0)) if step else np.array([],int)
        if len(nn):
            visit=visits[nn,streets[nn],actors[nn]];assert (visit<128).all()
            x=templates[ri[nn],rep[nn],streets[nn],actors[nn]].copy();x[:,20:]=dynamic[nn]
            pp=action.predict_proba(x,thread_count=2);pp=np.maximum(pp,1e-12)*legal[nn];pp/=pp.sum(1)[:,None]
            random=u[rep[nn],streets[nn],actors[nn],visit]
            kk=(random[:,0,None]>pp.cumsum(1)).sum(1).clip(0,3)
            kinds[nn]=kk;amounts[nn]=np.where(kk==2,amounts[nn],0);rp=np.flatnonzero(kk==3)
            if len(rp):
                raises=nn[rp];sp=sizer.predict_proba(x[rp],thread_count=2)
                chosen=(random[rp,1,None]>sp.cumsum(1)).sum(1).clip(0,sp.shape[1]-1)
                for i,k in zip(raises,chosen):
                    call,remaining,pot,minimum=params[i]
                    want=remaining if k==len(centers)-1 else np.floor(max(0.,np.exp(centers[k])-.01)*max(pot,1.))
                    amounts[i]=min(remaining,max(want,call+minimum))
        if step==0:kinds=forced_k;amounts=forced_amount
        assert n.LIB.apply_batch(states,count,actors,kinds,amounts)==0
                                                                              
        visits[ids,streets[ids],actors[ids]]+=1;total+=len(ids)
    else:raise RuntimeError('128-action rollout limit; no truncated values exported')
    assert int(visits.sum())==total
    out=np.empty((count,6));assert n.LIB.payouts(states,count,ranks,out)==0
    error=float(abs(out.sum(1)).max());assert error<1e-8
    return out.reshape(len(roots),2,reps,6),dict(trajectories=count,actions=total,max_steps=maxstep,chip_conservation_error=error)

def main():
    ROOT.mkdir(exist_ok=True);q=pl.read_parquet('artifacts/evidence_session107_rollout_precision/selection.parquet')
    rows=[];audits=[];start=time.time();original=v.native.simulate
    for (table,),local in q.group_by('table_id',maintain_order=True):
        native=int(local['fold'][0]);roots=v.s.roots_for_table(table,local,256);action,size,meta=v.s.models(native,(native+1)%4)
        old,oi,ov=v.evaluate(roots,action,size,meta,256)
        try:
            v.native.simulate=simulate;new,info,nv=v.evaluate(roots,action,size,meta,256)
            if not audits:
                _,_,prefix=v.evaluate(roots,action,size,meta,32);np.testing.assert_array_equal(prefix,nv[:,:,:32])
                rev,_,rv=v.evaluate(roots[::-1],action,size,meta,256);np.testing.assert_array_equal(nv,rv[::-1])
        finally:v.native.simulate=original
                                                                          
        np.testing.assert_array_equal(ov[:,2:],nv[:,2:])
        for i,r in enumerate(roots):
            for member in r['members']:
                own,other=r['own'],member['other'];scale=max(1.,r['pot'])
                x=(ov[i,0]-ov[i,1])/scale;y=(nv[i,0]-nv[i,1])/scale
                x=np.column_stack([-x[:,own],x[:,other],x[:,own]+x[:,other]])
                y=np.column_stack([-y[:,own],y[:,other],y[:,own]+y[:,other]])
                for j,field in enumerate(['own_loss','partner_gain','team_gain']):
                    rows.append(dict(query_id=member['query_id'],street=r['state'].street,field=field,
                        step_mean=float(x[:,j].mean()),actor_mean=float(y[:,j].mean()),step_variance=float(x[:,j].var(ddof=1)),actor_variance=float(y[:,j].var(ddof=1))))
        audits.append(dict(table=table,roots=len(roots),checkcall_parity_error=0,**info));print('COUPLING_POOL',table,time.time()-start,flush=True)
    z=pl.DataFrame(rows).with_columns(pl.when(pl.col('step_variance')>1e-20).then(pl.col('actor_variance')/pl.col('step_variance')).otherwise(None).alias('ratio'))
    z.write_parquet(ROOT/'precision.parquet')
    report=dict(method=__doc__,pools=len(audits),roots=sum(x['roots'] for x in audits),seconds=time.time()-start,
        summaries=z.group_by('street','field').agg(pl.col('ratio').median().alias('median_variance_ratio'),pl.col('step_variance').mean(),pl.col('actor_variance').mean()).sort('street','field').to_dicts(),
        overall_variance_ratio=float(z['actor_variance'].sum()/z['step_variance'].sum()),query_reversal_error=0,replicate_prefix_error=0,
        counter_accounting='Total per-state visit counters equal total applied actions; each street/actor count only increments.',
        caveat='Finite precision pilot on81 label-free selected roots. Different common-random-number coupling changes empirical finite-sample means; no evidence performance is measured.')
    (ROOT/'pilot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
