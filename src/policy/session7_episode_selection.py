\
\
\
\
\
import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from session6_priority import inclusion
ROOT=Path('artifacts/evidence_session7')
def parameters(a,b,t,tau,contrast=.75):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
    scale=np.maximum(1,a+b);a=a/scale;b=b/scale;s=a+b;delta=contrast*np.minimum(s,1-s);presence=np.stack([s-delta,s+delta],1);ratio=a/np.maximum(s,1e-15);p1=presence*ratio[:,None];p2=presence*(1-ratio[:,None]);p=np.stack([1-presence,p1,p2],2)
    r=np.exp(-np.diff(t)/tau) if tau>0 else np.zeros(len(t)-1);T=np.array([[[.5+.5*v,.5-.5*v],[.5-.5*v,.5+.5*v]] for v in r]);return p,T
def correlated_inclusion(a,b,t,tau,contrast=.75,cap=5):
    p,T=parameters(a,b,t,tau,contrast);n=len(a)
    def forward(count):
        z=np.zeros((n,cap,2));z[0,0,:]=.5
        for i in range(n-1):
            after=z[i]*(1-count[i]);after[1:]+=z[i,:-1]*count[i];z[i+1]=after@T[i]
        return z
    primary=p[:,:,1];secondary=p[:,:,2];fp=forward(primary);fa=forward(primary+secondary);back=np.zeros((n,cap,2));back[-1,0,:]=1
    for i in range(n-2,-1,-1):
        future=back[i+1]*(1-primary[i+1]);future[1:]+=back[i+1,:-1]*primary[i+1];back[i]=future@T[i].T
    out=np.empty(n)
    for i in range(n):out[i]=np.sum(fp[i]*primary[i])+sum(np.sum(fa[i,k]*secondary[i]*back[i,:cap-k].sum(0)) for k in range(cap))
    return out
def check():
    rng=np.random.default_rng(7121);a=rng.uniform(.02,.6,6);b=rng.uniform(.02,.4,6);t=np.arange(6)*40.;p,T=parameters(a,b,t,100);exact=np.zeros(6)
    for states in itertools.product(range(2),repeat=6):
        ps=.5*np.prod([T[i,states[i],states[i+1]] for i in range(5)])
        for categories in itertools.product(range(3),repeat=6):
            weight=ps*np.prod(p[np.arange(6),states,categories]);chosen=([i for i,c in enumerate(categories) if c==1]+[i for i,c in enumerate(categories) if c==2])[:5];exact[chosen]+=weight
    error=float(np.max(np.abs(exact-correlated_inclusion(a,b,t,100))));independent=float(np.max(np.abs(inclusion(a,b)-correlated_inclusion(a,b,t,0))));assert error<1e-12 and independent<1e-12
    (ROOT/'episode_selection_checks.json').write_text(json.dumps({'enumerated_hidden_state_and_event_paths':6**6,'max_error':error,'independent_limit_error':independent},indent=2));print('episode checks',error,independent,flush=True)
def main():
    check();d=pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet');parts={100:[],400:[]}
    for _,g in d.sort('pair_id','time','hand_id').group_by('pair_id',maintain_order=True):
        a=g['primary'].to_numpy();b=g['secondary'].to_numpy();t=g['time'].to_numpy()*5000
        for tau in parts:
            s=correlated_inclusion(a,b,t,tau);assert s.min()>=0 and s.max()<=1+1e-10 and s.sum()<=5+1e-10;parts[tau].append(g.select('pair_id','hand_id').with_columns(pl.Series('score',s)))
    for tau,rows in parts.items():pl.concat(rows).write_parquet(ROOT/f'episode_select{tau}_oof.parquet')
if __name__=='__main__':main()
