\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session11_conditional_family import Model,features,template,log_count_at_least
from session8_data import hand_data
from session8_count_conditioning import conditioned
ROOT=Path('artifacts/evidence_session12/type_mixture');C=pl.col;torch.set_num_threads(2)
CONFIG={'steps':120,'learning_rate':.03,'amplitude_bound':3.,'initial_amplitude':.5,'regularization_amplitude':.001,'regularization_bias':.01,'regularization_gate':.01,'gating_features':'six label-free summaries of frozen event/type probabilities interacted with three families plus family indicators; train-only normalization','control':'same emissions and gate, latent subtype resampled per hand','model':'one subtype state shared by whole relationship','event_rates':'fixed to frozen R30; only primary/fallback allocation varies'}
def summaries(p,mask,families):
 out=[]
 for q,m,f in zip(p,mask,families):
  q=q[m];s=q[:,1:].sum(1);r=q[:,1]/np.maximum(s,1e-15);w=s/max(s.sum(),1e-15);z=np.clip(np.log(np.maximum(r,1e-6))-np.log(np.maximum(1-r,1e-6)),-6,6);order=np.argsort(-s)[:5];half=max(1,len(q)//2);v=np.array([(w*r).sum(),(w*z).sum(),np.sqrt((w*(z-(w*z).sum())**2).sum()),r[order].mean(),(s[:half]*r[:half]).sum()/max(s[:half].sum(),1e-15),(s[half:]*r[half:]).sum()/max(s[half:].sum(),1e-15)]);x=np.zeros(21);x[int(f)*6:int(f)*6+6]=v;x[18+int(f)]=1;out.append(x)
 return np.array(out)
class TypeMixture(nn.Module):
 def __init__(self):
  super().__init__();self.gate=nn.Linear(21,1,dtype=torch.float64);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias);self.amp=nn.Parameter(torch.full((3,),-np.log(5),dtype=torch.float64));self.bias=nn.Parameter(torch.zeros(3,dtype=torch.float64))
 def forward(self,p,x,f):
  event=p[:,:,1:].sum(2);r=p[:,:,1]/event.clamp_min(1e-30);odds=torch.log(r.clamp_min(1e-30))-torch.log((1-r).clamp_min(1e-30));shift=self.bias[f,None,None]+3*torch.sigmoid(self.amp[f,None,None])*torch.tensor([-1.,1.])[None,None];rr=torch.sigmoid(odds[:,:,None]+shift);primary=event[:,:,None]*rr;prob=torch.stack([p[:,:,0,None].expand_as(primary),primary,event[:,:,None]-primary],3);gate=torch.sigmoid(self.gate(x)).squeeze(1);pi=torch.stack([1-gate,gate],1);return prob,pi

def loglist(prob,tt,v):
 lp=torch.log(prob.clamp_min(1e-100));terms=torch.stack([torch.zeros_like(lp[:,:,0]),lp[:,:,1],lp[:,:,2],lp[:,:,0],torch.logsumexp(lp[:,:,[0,2]],2)],2);val=terms[:,None].expand(-1,tt.shape[1],-1,-1).gather(3,tt[:,:,:,None]).squeeze(3).sum(2);return torch.logsumexp(val.masked_fill(~v,-1e6),1)
def objective(prob,pi,tt,v,den,normal,kind):
 if kind=='independent':lp=loglist((prob*pi[:,None,:,None]).sum(2),tt,v)
 else:lp=torch.logsumexp(torch.stack([loglist(prob[:,:,s],tt,v) for s in range(2)],1)+pi.log(),1)
 return (-lp+normal)/den

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();parts=[];audit=[];start=time.time()
 for f in range(4):
  q=d.join(pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');bags=[]
  for (pid,),g in q.group_by('pair_id'):
   g=g.sort('time','hand_id');x,p=features(g);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];bags.append({'pid':pid,'g':g,'x':x,'p':p,'template':template(len(g),e),'den':len(e),'fold':g['fold'][0],'family':['directed_transfer','soft_play','coordinated_isolation'].index(g['behavior_family'][0])})
  bags.sort(key=lambda b:b['pid']);N=len(bags);n=max(len(b['g']) for b in bags);X=np.zeros((N,n,35),np.float32);P=np.zeros((N,n,2),np.float32);M=np.zeros((N,n),bool);TT=np.zeros((N,6,n),np.int64);V=np.zeros((N,6),bool);D=np.array([b['den'] for b in bags]);F=np.array([b['family'] for b in bags]);fv=np.array([b['fold'] for b in bags])
  for i,b in enumerate(bags):
   k=len(b['g']);X[i,:k]=b['x'];P[i,:k]=b['p'];M[i,:k]=True;t=b['template'];TT[i,:len(t),:k]=t;V[i,:len(t)]=True
  tr=np.flatnonzero((fv!=f)&V.any(1));va=np.flatnonzero(fv==f);probs=[];normals=[];feats=[]
  for seed in [1010,2020]:
   st=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);base=Model(35,'independent');base.load_state_dict(st['state_dict']);base.eval();K=np.array([st['minimums'][b['g']['behavior_family'][0]] for b in bags])
   with torch.no_grad():
    prior=(torch.tensor(P)+base(torch.tensor(np.clip((X-st['mu'])/st['sd'],-6,6)),torch.tensor(M))).double();p=torch.softmax(torch.cat([torch.zeros_like(prior[:,:,:1]),prior],2),2);p[~torch.tensor(M)]=torch.tensor([1.,0.,0.],dtype=torch.float64);normal=log_count_at_least(prior,torch.tensor(M),torch.tensor(K));probs.append(p);normals.append(normal);feats.append(summaries(p.numpy(),M,F))
  mu=np.concatenate([x[tr] for x in feats]).mean(0);sd=np.maximum(.05,np.concatenate([x[tr] for x in feats]).std(0));XX=[torch.tensor(np.clip((x-mu)/sd,-6,6)) for x in feats];pp=torch.cat([p[tr] for p in probs]);xx=torch.cat([x[tr] for x in XX]);idx=np.tile(tr,2);tt=torch.tensor(TT[idx]);valid=torch.tensor(V[idx]);den=torch.tensor(D[idx]);family=torch.tensor(F[idx]);normal=torch.cat([z[tr] for z in normals]);pred={}
  for kind in ['independent','shared']:
   m=TypeMixture();opt=torch.optim.Adam(m.parameters(),lr=.03);trace=[]
   for step in range(120):
    p,pi=m(pp,xx,family);ll=objective(p,pi,tt,valid,den,normal,kind).mean();reg=.001*(3*torch.sigmoid(m.amp)).square().mean()+.01*m.bias.square().mean()+.01*m.gate.weight.square().mean();loss=ll+reg;assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();trace.append(float(loss.detach()))
   torch.save({'state_dict':m.state_dict(),'fold':f,'kind':kind,'mu':mu,'sd':sd,'minimums':st['minimums'],'config':CONFIG},ROOT/f'{kind}_fold{f}.pt');outputs=[]
   with torch.no_grad():
    for p,x in zip(probs,XX):
     v,pi=m(p[va],x[va],torch.tensor(F[va]));assert torch.max(abs(v[:,:,:,1:].sum(3)-p[va,:,1:].sum(2)[:,:,None]))<1e-12;outputs.append((v.numpy(),pi.numpy()))
   pred[kind]=outputs;audit.append({'fold':f,'kind':kind,'training_pairs':len(tr),'loss_trace':trace,'amplitude':(3*torch.sigmoid(m.amp)).detach().tolist(),'type_bias':m.bias.detach().tolist()});print('type mixture',f,kind,round(time.time()-start,1),flush=True)
  for j,i in enumerate(va):
   g=bags[i]['g'];n0=len(g);out=g.select('pair_id','hand_id')
   for kind,outputs in pred.items():
    values=[]
    for p,pi in outputs:
     z=p[j,:n0];w=pi[j]
     if kind=='independent':v=conditioned((z*w[None,:,None]).sum(1)[:,1:],K[i])
     else:v=sum(w[s]*conditioned(z[:,s,1:],K[i]) for s in range(2))
     values.append(v)
    score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*np.mean(values,0);out=out.with_columns(pl.Series(kind,score))
   parts.append(out)
  (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
 pl.concat(parts).write_parquet(ROOT/'type_mixture_oof.parquet')
if __name__=='__main__':main()
