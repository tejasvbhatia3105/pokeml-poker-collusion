\
\
\
\
\
import ctypes,hashlib,json,time
from pathlib import Path
import numpy as np
import polars as pl
import session108_action_frontier as f

ROOT=Path('artifacts/evidence_session111_native_simulator')
LIB=ctypes.CDLL(str((ROOT/'engine.dylib').resolve()))
D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
F=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
B=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
R=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
LIB.state_width.restype=ctypes.c_int;assert LIB.state_width()==45
LIB.decisions.argtypes=[D,ctypes.c_int,I];LIB.decisions.restype=None
LIB.dynamic_features.argtypes=[D,ctypes.c_int,I,F,B,D];LIB.dynamic_features.restype=None
LIB.apply_batch.argtypes=[D,ctypes.c_int,I,I,D];LIB.apply_batch.restype=ctypes.c_int
LIB.payouts.argtypes=[D,ctypes.c_int,R,D];LIB.payouts.restype=ctypes.c_int

def provenance():
    paths=[Path(__file__),Path('src/policy/session111_native_engine.cpp'),ROOT/'engine.dylib']
    return {str(p.resolve().relative_to(Path.cwd())):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}

def state(st):
    return np.r_[st.starting,st.contribution,st.committed,st.alive,st.pending,st.raise_right,
        st.minimum_raise,st.button,st.cursor,st.action_no,st.previous_action,st.last_aggressor,st.prior_raises,st.street,st.big_blind].astype(np.float64)

def simulate(roots,action,sizer,meta,reps=64):
    n=len(roots)*2*reps;states=np.repeat(np.stack([state(r['state']) for r in roots]),2*reps,axis=0)
    ri=np.repeat(np.arange(len(roots)),2*reps);sc=np.tile(np.repeat(np.arange(2),reps),len(roots));rep=np.tile(np.arange(reps),2*len(roots))
    templates=np.stack([r['templates'][:reps] for r in roots]);ranks=np.ascontiguousarray(np.stack([r['ranks'][:reps] for r in roots])[ri,rep],dtype=np.uint32)
    u=np.random.default_rng(10401).random((reps,f.s.CONFIG['max_actions'],2));centers=np.array(meta['centers'])
    actors=np.empty(n,np.int32);dynamic=np.zeros((n,13),np.float32);legal=np.zeros((n,4),np.uint8);params=np.zeros((n,4))
    forced_k=np.array([r['forced_class'] for r in roots],np.int32)[ri];forced_amount=np.array([r['forced_amount'] for r in roots],np.float64)[ri];total=0;maxstep=0
    for step in range(f.s.CONFIG['max_actions']):
        LIB.decisions(states,n,actors);active=actors>=0
        if not active.any():break
        maxstep=step+1;LIB.dynamic_features(states,n,actors,dynamic,legal,params)
        kinds=np.where(params[:,0]>0,2,1).astype(np.int32);amounts=params[:,0].copy()
        nn=np.flatnonzero(active&(sc==0)) if step else np.array([],int)
        if len(nn):
            x=templates[ri[nn],rep[nn],states[nn,43].astype(int),actors[nn]].copy();x[:,20:]=dynamic[nn]
            pp=action.predict_proba(x,thread_count=2);pp=np.maximum(pp,1e-12)*legal[nn];pp/=pp.sum(1)[:,None]
            kk=(u[rep[nn],step,0,None]>pp.cumsum(1)).sum(1).clip(0,3)
            kinds[nn]=kk;amounts[nn]=np.where(kk==2,amounts[nn],0);rp=np.flatnonzero(kk==3)
            if len(rp):
                raises=nn[rp];sp=sizer.predict_proba(x[rp],thread_count=2)
                chosen=(u[rep[raises],step,1,None]>sp.cumsum(1)).sum(1).clip(0,sp.shape[1]-1)
                for i,k in zip(raises,chosen):
                    call,remaining,pot,minimum=params[i]
                    want=remaining if k==len(centers)-1 else np.floor(max(0.,np.exp(centers[k])-.01)*max(pot,1.))
                    amounts[i]=min(remaining,max(want,call+minimum))
        if step==0:kinds=forced_k;amounts=forced_amount
        error=LIB.apply_batch(states,n,actors,kinds,amounts);assert error==0,error
        total+=int(active.sum())
    else:raise RuntimeError('128-action rollout limit; no truncated values exported')
    out=np.empty((n,6));assert LIB.payouts(states,n,ranks,out)==0
    error=float(abs(out.sum(1)).max());assert error<1e-8
    return out.reshape(len(roots),2,reps,6),dict(trajectories=n,actions=total,max_steps=maxstep,chip_conservation_error=error)

def pilot():
    q=pl.read_parquet(f.ROOT/'queries.parquet');records=[];original=f.simulate
    for native in range(4):
        table=q.filter(pl.col('fold')==native)['table_id'].unique().sort()[0]
        local=q.filter(pl.col('table_id')==table);keys=[]
        for street in range(4):
            z=local.filter(pl.col('street_no')==street).select('hand_id','action_no').unique().sort('hand_id','action_no').head(2)
            if len(z):keys.append(z)
        local=local.join(pl.concat(keys),on=['hand_id','action_no'],how='semi');roots=f.s.roots_for_table(table,local,64)
        for outer in range(4):
            if native==outer:continue
            action,size,meta=f.s.models(native,outer);start=time.time();want,info,v,offset=f.evaluate(roots,action,size,meta,64);oldseconds=time.time()-start
            try:
                f.simulate=simulate;start=time.time();got,newinfo,nv,noffset=f.evaluate(roots,action,size,meta,64);seconds=time.time()-start
                reverse,_,_,_=f.evaluate(roots[::-1],action,size,meta,64)
            finally:f.simulate=original
            np.testing.assert_array_equal(v,nv);np.testing.assert_array_equal(want.to_numpy(),got.to_numpy());assert offset==noffset and info==newinfo
            np.testing.assert_array_equal(got.sort('query_id').to_numpy(),reverse.sort('query_id').to_numpy())
            prefix,_=simulate(roots,action,size,meta,16);old,_=original(roots,action,size,meta,64);np.testing.assert_array_equal(prefix,old[:,:,:16])
            records.append(dict(table=table,native=native,outer=outer,roots=len(roots),python_seconds=oldseconds,native_seconds=seconds,
                every_path_payoff_error=0,feature_error=0,query_reversal_error=0,replicate_prefix_error=0,**info))
            print('NATIVE_PARITY',native,outer,'speedup',oldseconds/seconds,flush=True)
    report=dict(provenance=provenance(),records=records,reference_blocks=len(records),
        trajectories=sum(r['trajectories'] for r in records),python_seconds=sum(r['python_seconds'] for r in records),native_seconds=sum(r['native_seconds'] for r in records))
    (ROOT/'pilot.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':pilot()
