\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import json
from pathlib import Path
import numpy as np
import torch
from session193_selection_marginals import selected_probabilities

ROOT=Path('artifacts/evidence_session194_selected_map')
RANK_TEMPERATURE=.25
CUT_TEMPERATURE=.5

def smooth_ap(score,y,mask,den,rank_temperature=RANK_TEMPERATURE,cut_temperature=CUT_TEMPERATURE):
                                                                        
    ix=torch.topk(y,5,dim=1).indices
    keep=y.gather(1,ix)
    z=torch.logit(score.clamp(1e-7,1-1e-7))
    pos=z.gather(1,ix)
    outrank=torch.sigmoid((z[:,None,:]-pos[:,:,None])/rank_temperature)*mask[:,None,:]
    rank=.5+outrank.sum(2)                                               
    positives=.5+(outrank*y[:,None,:]).sum(2)
    top=torch.sigmoid((5.5-rank)/cut_temperature)
    return ((positives/rank)*top*keep).sum(1)/den

def objective(P,A,M,Y,D,train,K,fixed):
    score=fixed[train]+.5*selected_probabilities(P[train]+A[train],M[train],K[train])
    ap=smooth_ap(score,Y[train],M[train],D[train])
    reg=(A[train].square().sum(2)*M[train]).sum(1)/M[train].sum(1)
    return (1-ap+.02*reg).sum()

def check():
    rng=np.random.default_rng(194);p=torch.tensor(rng.uniform(.01,.99,(4,12)),requires_grad=True)
    y=torch.zeros((4,12),dtype=torch.float64)
    for j in range(4):y[j,rng.choice(12,3+j%3,replace=False)]=1
    mask=torch.ones_like(y,dtype=torch.bool);den=y.sum(1)
    assert torch.autograd.gradcheck(lambda s:smooth_ap(s,y,mask,den),(p,),eps=1e-6,atol=1e-5)
    near=smooth_ap(p,y,mask,den,1e-5,1e-5).detach().numpy();exact=[]
    for ss,yy in zip(p.detach().numpy(),y.numpy()):
        yy=yy[np.argsort(-ss)[:5]];exact.append(float((yy*yy.cumsum()/np.arange(1,6)).sum()))
    exact=np.array(exact)/den.numpy();err=float(abs(near-exact).max());assert err<1e-10
                                                                                 
    z=torch.tensor(rng.normal(-2,1,(2,7,2)),requires_grad=True)
    m=torch.ones((2,7),dtype=torch.bool);yy=y[:2,:7].clone();yy[:]=0;yy[:,[0,2,5]]=1
    fixed=torch.tensor(rng.uniform(.01,.3,(2,7)))
    fn=lambda zz:smooth_ap(fixed+.5*selected_probabilities(zz,m,torch.tensor([3,3])),yy,m,torch.tensor([3.,3.]))
    assert torch.autograd.gradcheck(fn,(z,),eps=1e-6,atol=1e-5)
    ROOT.mkdir(exist_ok=True)
    r=dict(method=__doc__,smooth_rank_gradcheck=True,through_selection_gradcheck=True,
           zero_temperature_exact_MAP_error=err,rank_temperature=RANK_TEMPERATURE,cut_temperature=CUT_TEMPERATURE)
    (ROOT/'math_verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

def train():
    import session193_selected_membership_training as t
    t.ROOT=ROOT;t.objective=objective
    t.CONFIG=dict(t.CONFIG,method=__doc__,objective='one minus smooth AP5 of final selected score + pair-balanced residual penalty',
                  rank_temperature=RANK_TEMPERATURE,cut_temperature=CUT_TEMPERATURE)
    t.main()
    p=ROOT/'report.json';r=json.loads(p.read_text());r['method']=__doc__
    r['arm_names']={'compact_control':'matched190_joint_list','action_list':'194_selected_smoothMAP',
                    'compact_pressure':'matched190_with_pressure59','action_pressure':'194_with_pressure59'}
    p.write_text(json.dumps(r,indent=2))

if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='train':train()
    else:check()
