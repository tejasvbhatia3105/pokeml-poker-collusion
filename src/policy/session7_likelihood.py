\
\
\
\
\
import itertools
import numpy as np
def masks(n,ordered_indices,cap=5):
    e=np.asarray(ordered_indices,dtype=int);m=len(e);out=[]
    assert m<=cap and len(set(e))==m
    for k in range(m+1):
        primary=e[:k];secondary=e[k:]
        if np.any(np.diff(primary)<=0) or np.any(np.diff(secondary)<=0):continue
        z=np.zeros((n,3),bool);z[:,0]=True
        if m==cap:
            if k==cap:
                z[:,2]=True;z[primary[-1]+1:,:]=True
            else:z[secondary[-1]+1:,2]=True
        z[e,:]=False;z[primary,1]=True;z[secondary,2]=True
        out.append((k,z))
    return out
def posterior(prob,ordered_indices):
    hypotheses=masks(len(prob),ordered_indices) if prob.shape[1]==3 else multilevel_masks(len(prob),ordered_indices,prob.shape[1]-1)
    if not hypotheses:return None,None,None
    p=np.maximum(np.asarray(prob,float),1e-12);p/=p.sum(1,keepdims=True)
    mass=np.stack([(p*mask).sum(1) for _,mask in hypotheses])
    ll=np.log(mass).sum(1);mx=ll.max();w=np.exp(ll-mx);w/=w.sum()
    expected=sum(weight*p*mask/m[:,None] for weight,m,(_,mask) in zip(w,mass,hypotheses))
    assert np.allclose(expected.sum(1),1)
    return expected,float(mx+np.log(np.exp(ll-mx).sum())),dict(zip([k for k,_ in hypotheses],w.tolist()))
def multilevel_masks(n,ordered_indices,levels,cap=5):
    e=np.asarray(ordered_indices,dtype=int);out=[];m=len(e)
    for classes in itertools.combinations_with_replacement(range(1,levels+1),m):
        c=np.array(classes,dtype=int);parts=[e[c==k] for k in range(1,levels+1)]
        if any(np.any(np.diff(part)<=0) for part in parts):continue
        z=np.zeros((n,levels+1),bool);z[:,0]=True
        if m==cap:
            worst=classes[-1];z[:,worst+1:]=True;z[e[-1]+1:,worst]=True
        z[e,:]=False;z[e,c]=True
        out.append((','.join(str(len(part)) for part in parts),z))
    return out
def multilevel_inclusion(prob,cap=5):
    \
    n,classes=prob.shape;out=np.zeros(n);higher=np.zeros(n)
    def scan(p):
        z=np.zeros((len(p)+1,cap));z[0,0]=1
        for i,v in enumerate(p):z[i+1]=np.r_[z[i,0]*(1-v),z[i,1:]*(1-v)+z[i,:-1]*v]
        return z
    for k in range(1,classes):
        same=prob[:,k];pre=scan(higher+same);suf=scan(higher[::-1])[::-1]
        for i in range(n):out[i]+=same[i]*sum(pre[i,j]*suf[i+1,:cap-j].sum() for j in range(cap))
        higher+=same
    return out
