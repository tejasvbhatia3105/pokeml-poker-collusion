import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
import numpy as np,polars as pl,joblib,torch
from scipy.special import softmax
from session6_priority import inclusion
import session134_rate_likelihood as variant
import session130_hierarchical_list as h

def nodes(n):
    x,w=np.polynomial.hermite.hermgauss(n);h.QX=x*np.sqrt(2);h.QW=w/np.sqrt(np.pi)

def stable_inclusion(z,k,sigma):
                                                                               
                                                                             
                                                                             
    pp=softmax(np.concatenate([np.zeros((len(h.QX),len(z),1)),z[None]+h.QX[:,None,None]*sigma],2),axis=2)
    s=pp[:,:,1:].sum(2);Q,N=s.shape
    pre=np.zeros((Q,N+1,k));suf=np.zeros_like(pre);pre[:,0,0]=1;suf[:,N,0]=1
    for i in range(N):
        pre[:,i+1,0]=pre[:,i,0]*(1-s[:,i])
        pre[:,i+1,1:]=pre[:,i,1:]*(1-s[:,i,None])+pre[:,i,:-1]*s[:,i,None]
        j=N-1-i;suf[:,j,0]=suf[:,j+1,0]*(1-s[:,j])
        suf[:,j,1:]=suf[:,j+1,1:]*(1-s[:,j,None])+suf[:,j+1,:-1]*s[:,j,None]
    low=np.zeros_like(s)
    for j in range(k-1):
        for l in range(k-1-j):low+=pre[:,:-1,j]*suf[:,1:,l]
    low*=s
    ordinary=np.stack([inclusion(p[:,1],p[:,2]) for p in pp]);joint=ordinary-low
    assert joint.min()>-1e-10
    logits=torch.tensor(np.log(pp[:,:,1:]/pp[:,:,:1]),dtype=torch.float64)
    tail=h.log_count_at_least(logits,torch.ones(logits.shape[:2],dtype=torch.bool),torch.full((Q,),k)).numpy()
    result=h.QW@joint.clip(0)/(h.QW@np.exp(tail))
    assert result.min()>=0 and result.max()<1+1e-10
    return result.clip(0,1)

def main():
    variant.setup()
    original_inclusion=h.inclusion_mixture;root=h.ROOT
    h.inclusion_mixture=stable_inclusion;h.ROOT=root/'stable_math';h.ROOT.mkdir(exist_ok=True);h.check();h.ROOT=root;h.inclusion_mixture=original_inclusion
    d=h.data();records=[];parts=[];replay_error=0.
    for f in range(4):
        (groups,x,big,P,M,T,V,D,fv,bid,pos),fam,K,tr=h.pack(d,f)
        X=np.nan_to_num(np.column_stack([x,big]),nan=0,posinf=1e6,neginf=-1e6)
        model=joblib.load(h.ROOT/f'model_random_rate_fold{f}.joblib');a=h.infer(model,X)
        for i in np.flatnonzero(fv==f):
            g=groups[i];z=P[i,:len(g)].double().numpy()+a[bid==i];sd=float(model['sigma'][fam[i]]);out={}
            for n in [17,33,65]:
                nodes(n);out[n]=stable_inclusion(z,int(K[i]),sd)
            nodes(17);replay_error=max(replay_error,float(abs(out[17]-original_inclusion(z,int(K[i]),sd)).max()))
            records.append(dict(pair_id=g['pair_id'][0],fold=f,sigma=sd,max17to65=float(abs(out[17]-out[65]).max()),max33to65=float(abs(out[33]-out[65]).max())))
            parts.append(g.select('pair_id','hand_id').with_columns(*[pl.Series('inclusion_q'+str(n),out[n]) for n in out]))
        print('quadrature fold',f,'sigma',model['sigma'].tolist(),flush=True)
    nodes(17);pl.DataFrame(records).write_parquet(h.ROOT/'quadrature_pairs.parquet');pl.concat(parts).write_parquet(h.ROOT/'quadrature_inclusion.parquet')
    report=dict(heldout_labels_used=False,pairs=len(records),max17to65=max(r['max17to65'] for r in records),max33to65=max(r['max33to65'] for r in records),max_sigma=max(r['sigma'] for r in records),stable_vs_original17_max_error=replay_error)
    assert replay_error<1e-10
    report['criterion_passed']=report['max17to65']<=.001 and report['max33to65']<=1e-5
    report['criteria']='max17to65<=.001 and max33to65<=1e-5; otherwise numerical convergence unresolved'
    (h.ROOT/'quadrature.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
