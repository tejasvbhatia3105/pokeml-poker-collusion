\
\
\
\
\
import itertools,json
from pathlib import Path
import numpy as np
ROOT=Path('artifacts/evidence_session162_action_em_math')

def normalize(p):
    p=np.maximum(np.asarray(p,dtype=np.float64),1e-12)
    return p/p.sum(1,keepdims=True)

def hand_probabilities(p,bag,n):
    p=normalize(p);none=np.zeros(n);not_primary=np.zeros(n)
    np.add.at(none,bag,np.log(p[:,0]));np.add.at(not_primary,bag,np.log1p(-p[:,1]))
    h=np.column_stack([np.exp(none),-np.expm1(not_primary),np.exp(not_primary)*(-np.expm1(none-not_primary))])
    assert np.isfinite(h).all() and h.min()>=-1e-12 and np.allclose(h.sum(1),1)
    return np.maximum(h,0),none,not_primary

def responsibilities(p,bag,hand_target):
    p=normalize(p);h,ln0,lnnp=hand_probabilities(p,bag,len(hand_target))
    other_np=lnnp[bag]-np.log1p(-p[:,1]);other_0=ln0[bag]-np.log(p[:,0])
    any_primary=-np.expm1(np.minimum(other_np,0))
    secondary_without_primary=np.exp(other_np)*(-np.expm1(np.minimum(other_0-other_np,0)))
    w=np.asarray(hand_target)[bag];a=w[:,1]/np.maximum(h[bag,1],1e-300);b=w[:,2]/np.maximum(h[bag,2],1e-300)
    target=np.column_stack([w[:,0]+a*p[:,0]*any_primary+b*p[:,0]*secondary_without_primary,
                           a*p[:,1],a*p[:,2]*any_primary+b*p[:,2]*np.exp(other_np)])
    assert np.isfinite(target).all() and target.min()>=-1e-10
    np.testing.assert_allclose(target.sum(1),w.sum(1),rtol=1e-8,atol=1e-9)
    return np.maximum(target,0)

def initialize(hand_prior,bag):
    counts=np.bincount(bag,minlength=len(hand_prior));assert (counts>0).all()
    h=normalize(hand_prior);a0=np.exp(np.log(h[:,0])/counts);anp=np.exp(np.log1p(-h[:,1])/counts)
    return np.column_stack([a0,1-anp,anp-a0])[bag]

def verify():
    rng=np.random.default_rng(16201);checks=[]
    for n in range(1,7):
        for trial in range(4):
            p=rng.dirichlet([2,.7,.7],n);bag=np.zeros(n,dtype=int);w=rng.dirichlet([1,1,1])[None]
            mass=np.zeros(3);joint=np.zeros((3,n,3))
            for state in itertools.product(range(3),repeat=n):
                k=1 if 1 in state else 2 if 2 in state else 0;prob=np.prod(p[np.arange(n),state]);mass[k]+=prob
                joint[k,np.arange(n),state]+=prob
            hp=hand_probabilities(p,bag,1)[0][0]
            expected=(joint/mass[:,None,None]*w.reshape(3,1,1)).sum(0)
            actual=responsibilities(p,bag,w)
            e=max(float(abs(hp-mass).max()),float(abs(actual-expected).max()));assert e<1e-10
            reverse=responsibilities(p[::-1],bag,w)[::-1];np.testing.assert_allclose(actual,reverse,atol=1e-12,rtol=0)
            init=initialize(w,bag);ie=float(abs(hand_probabilities(init,bag,1)[0]-w).max());assert ie<1e-10
            checks.append(dict(actions=n,trial=trial,enumeration_error=e,initial_marginal_error=ie))
    ROOT.mkdir(exist_ok=True);out=dict(checks=checks,scope='Exact enumeration for1–6 independent action categories, four draws each; responsibilities, permutation invariance and cold marginal initialization. No fitted model or performance claim.')
    (ROOT/'verification.json').write_text(json.dumps(out,indent=2));print('ACTION_EM_MATH_VERIFIED',len(checks),max(c['enumeration_error'] for c in checks))

if __name__=='__main__':verify()
