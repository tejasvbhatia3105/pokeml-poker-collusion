\
\
import numpy as np,torch
class ConditionalHMM(torch.autograd.Function):
 @staticmethod
 def forward(ctx,p,T,templates,valid,den,minimum):
  p=p.detach().numpy();T=T.detach().numpy();zcode=templates.numpy();valid=valid.numpy();den=den.numpy();minimum=minimum.numpy();B,n=p.shape[:2];S=6
  z=np.full((B,S,2),.5);ll=np.zeros((B,S));previous=[];transferred=[];scales=[];factors=[];codes=np.eye(3)[None,None]
                                                                              
  allowed=np.array([[0,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,1]],float)
  for i in range(n):
   previous.append(z);v=np.einsum('bks,bsr->bkr',z,T[:,i]);transferred.append(v);a=allowed[zcode[:,:,i]];factor=np.einsum('bkc,bsc->bks',a,p[:,i])+(zcode[:,:,i]==0)[:,:,None];factors.append(factor);after=v*factor;scale=np.maximum(after.sum(2),1e-100);scales.append(scale);ll+=np.log(scale);z=after/scale[:,:,None]
  ll=np.where(valid,ll,-1e6);peak=ll.max(1);ex=np.exp(ll-peak[:,None]);total=ex.sum(1);weights=ex/total[:,None];observed=peak+np.log(total);dp=np.zeros_like(p);dt=np.zeros_like(T);beta=np.ones_like(z)
  for i in range(n-1,-1,-1):
   df=weights[:,:,None]*transferred[i]*beta/scales[i][:,:,None];dp[:,i]-=np.einsum('bks,bkc->bsc',df,allowed[zcode[:,:,i]])/den[:,None,None];v=beta*factors[i]/scales[i][:,:,None];dt[:,i]-=np.einsum('bks,bkr,bk->bsr',previous[i],v,weights)/den[:,None,None];beta=np.einsum('bkr,bsr->bks',v,T[:,i])
  z=np.zeros((B,6,2));z[:,0]=.5;prev=[];trans=[]
  for i in range(n):
   prev.append(z);v=np.einsum('bks,bsr->bkr',z,T[:,i]);trans.append(v);s=p[:,i,:,1:].sum(2);z=v*p[:,i,None,:,0];z[:,1:5]+=v[:,:4]*s[:,None];z[:,5]=v[:,5]+v[:,4]*s
  beta=np.broadcast_to((np.arange(6)[None]>=minimum[:,None])[:,:,None],z.shape).astype(float);mass=np.maximum((z*beta).sum((1,2)),1e-100);factor=1/(mass*den);eventnext=np.array([1,2,3,4,5,5])
  for i in range(n-1,-1,-1):
   v=trans[i];s=p[:,i,:,1:].sum(2);bnext=beta[:,eventnext];d0=(v[:,:5]*beta[:,:5]).sum(1)*factor[:,None];d1=(v[:,:5]*bnext[:,:5]).sum(1)*factor[:,None];dp[:,i,:,0]+=d0;dp[:,i,:,1]+=d1;dp[:,i,:,2]+=d1;future=beta*p[:,i,None,:,0]+bnext*s[:,None];future[:,5]=beta[:,5];dt[:,i]+=np.einsum('bks,bkr->bsr',prev[i],future)*factor[:,None,None];beta=np.einsum('bkr,bsr->bks',future,T[:,i])
  ctx.save_for_backward(torch.tensor(dp),torch.tensor(dt));return torch.tensor((-observed+np.log(mass))/den)
 @staticmethod
 def backward(ctx,grad):
  dp,dt=ctx.saved_tensors;return dp*grad[:,None,None,None],dt*grad[:,None,None,None],None,None,None,None

def fast_loss(p,T,templates,valid,den,minimum):return ConditionalHMM.apply(p,T,templates,valid,den,minimum)

def check():
 from session12_learned_episode import loss,template
 rng=np.random.default_rng(1234);B=3;n=8;p=torch.tensor(rng.dirichlet([3,1,1],size=(B,n,2)),requires_grad=True);T=torch.tensor(rng.dirichlet([2,2],size=(B,n,2)),requires_grad=True);tt=np.zeros((B,6,n),int);valid=np.zeros((B,6),bool)
 for i,e in enumerate([np.array([0,2,4]),np.array([0,2,5,1,7]),np.array([1,2,3,4,5])]):
  z=template(n,e);tt[i,:len(z)]=z;valid[i,:len(z)]=True
 args=(torch.tensor(tt),torch.tensor(valid),torch.tensor([3,5,5]),torch.tensor([3,3,5]));slow=loss(p,T,*args);fast=fast_loss(p,T,*args);g1=torch.autograd.grad(slow.sum(),(p,T));g2=torch.autograd.grad(fast.sum(),(p,T));errors={'value_error':float(abs(slow-fast).max().detach()),'emission_gradient_error':float(abs(g1[0]-g2[0]).max()),'transition_gradient_error':float(abs(g1[1]-g2[1]).max())};print(errors,flush=True);assert max(errors.values())<1e-10;assert torch.autograd.gradcheck(lambda a,b:fast_loss(a,b,*args),(p,T),atol=1e-5);return errors
if __name__=='__main__':print(check())
