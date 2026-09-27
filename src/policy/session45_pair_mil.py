\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from scipy.special import logit,expit
from sklearn.metrics import average_precision_score
from session45_nested_pair import ROOT,PREV
torch.set_num_threads(2);C=pl.col
CONFIG={'seeds':[4511,4522],'epochs':50,'batch_size':128,'learning_rate':.001,'weight_decay':.03,'residual_l2':.01,'bound':2.,'selection':'fixed schedule, no validation checkpoint choice','arms':['tabular_only','action_bag'],'prior':'nested six-inner-block supervised GBDT risk; exact43 baseline on outer validation','loss':'binary cross entropy with original history weights plus residual square','caveat':'confirmed-label population; this is not actual R26 risk reproduction'}

def pack(d):
 cfg=json.load(open(PREV/'config.json'));bc=cfg['base_columns'];ac=pl.read_parquet(PREV/'action_features.parquet').sort('pair_id','time_index','hand_id');cols=[c for c in ac.columns if c not in ['pair_id','hand_id','time_index','table_id']];by={p:g for (p,),g in ac.group_by('pair_id')};counts=[];tokens=[]
 for pid,window in d.select('pair_id','window').iter_rows():
  lo,hi=cfg['windows'].get(window) or {'w0_1500':(0,1500),'w1500_3000':(1500,3000),'w500_2500':(500,2500),'w750_2250':(750,2250)}[window];g=by.get(pid);a=np.empty((0,len(cols))) if g is None else g.filter((C('time_index')>=lo)&(C('time_index')<hi)).select(cols).to_numpy();tokens.append(a);counts.append(len(a))
 n=max(1,max(counts));A=np.zeros((len(d),n,len(cols)),np.float32);M=np.zeros((len(d),n),bool)
 for i,a in enumerate(tokens):A[i,:len(a)]=a;M[i,:len(a)]=True
 return d.select(bc).to_numpy().astype(np.float32),A,M,bc,cols

class Model(nn.Module):
 def __init__(self,nx,na,kind):
  super().__init__();self.kind=kind;self.context=nn.Sequential(nn.Linear(nx,48),nn.GELU(),nn.Dropout(.1),nn.Linear(48,32),nn.GELU());self.action=nn.Sequential(nn.Linear(na,48),nn.GELU(),nn.Linear(48,32),nn.GELU());self.attention=nn.Linear(32,1);self.combine=nn.Sequential(nn.Linear(128,48),nn.GELU());self.head=nn.Linear(48,1);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x,a,mask):
  context=self.context(x)
  if self.kind=='tabular_only':pooled=torch.zeros((len(x),96),dtype=x.dtype,device=x.device)
  else:
   h=self.action(a);count=mask.sum(1,keepdim=True).clamp_min(1);mean=(h*mask[:,:,None]).sum(1)/count;maximum=h.masked_fill(~mask[:,:,None],-1e9).max(1).values;maximum=torch.where(mask.any(1,keepdim=True),maximum,torch.zeros_like(maximum));weights=torch.softmax(self.attention(h).squeeze(-1).masked_fill(~mask,-1e9),1)*mask;attended=(h*weights[:,:,None]).sum(1);pooled=torch.cat([mean,maximum,attended],1)
  return CONFIG['bound']*torch.tanh(self.head(self.combine(torch.cat([context,pooled],1))).squeeze(1))

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'model_config.json').write_text(json.dumps(CONFIG,indent=2));d=pl.read_parquet(PREV/'window_features.parquet');X,A,M,bc,ac=pack(d);fv=d['fold'].to_numpy();Y=torch.tensor(d['label'].to_numpy(),dtype=torch.float64);W=torch.tensor(np.where(d['window'].to_numpy()=='full',1,.5));pred={k:np.zeros(len(d)) for k in CONFIG['arms']};audit=[];start=time.time()
 for f in range(4):
  q=d.select('pair_id','window').join(pl.read_parquet(ROOT/f'nested_outer{f}.parquet'),on=['pair_id','window'],validate='1:1',maintain_order='left');prior=logit(q['risk'].to_numpy().clip(1e-10,1-1e-10));tr=fv!=f;va=fv==f;Z=np.column_stack([X,prior]).astype(np.float32);mu=Z[tr].mean(0);sd=np.maximum(.05,Z[tr].std(0));am=A[tr][M[tr]].mean(0);astd=np.maximum(.05,A[tr][M[tr]].std(0));xx=torch.tensor(np.clip((Z-mu)/sd,-8,8));aa=torch.tensor(np.clip((A-am)/astd,-8,8));mm=torch.tensor(M);pp=torch.tensor(prior);train=np.flatnonzero(tr);valid=np.flatnonzero(va)
  for kind in CONFIG['arms']:
   ensemble=[]
   for seed in CONFIG['seeds']:
    torch.manual_seed(seed+f);rng=np.random.default_rng(seed+f);model=Model(Z.shape[1],A.shape[2],kind);opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay']);trace=[]
    for epoch in range(CONFIG['epochs']):
     model.train();order=rng.permutation(train);losses=[]
     for j in range(0,len(order),CONFIG['batch_size']):
      ix=order[j:j+CONFIG['batch_size']];delta=model(xx[ix],aa[ix],mm[ix]);loss=(nn.functional.binary_cross_entropy_with_logits(pp[ix]+delta,Y[ix],reduction='none')*W[ix]).sum()/W[ix].sum()+CONFIG['residual_l2']*(delta.square()*W[ix]).sum()/W[ix].sum();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
     trace.append(float(np.mean(losses)))
    model.eval();chunks=[]
    with torch.no_grad():
     for j in range(0,len(valid),128):
      ix=valid[j:j+128];chunks.append(torch.sigmoid(pp[ix]+model(xx[ix],aa[ix],mm[ix])).numpy())
    ensemble.append(np.concatenate(chunks));torch.save({'state_dict':model.state_dict(),'kind':kind,'fold':f,'seed':seed,'mu':mu,'sd':sd,'action_mu':am,'action_sd':astd,'base_columns':bc,'action_columns':ac,'config':CONFIG},ROOT/f'{kind}_fold{f}_seed{seed}.pt');audit.append({'fold':f,'kind':kind,'seed':seed,'training_rows':len(train),'validation_rows':len(valid),'overlap':0,'training_loss':trace});print('pair MIL',f,kind,seed,round(time.time()-start,1),flush=True)
   pred[kind][va]=np.mean(ensemble,0)
 out=d.select('pair_id','table_id','fold','window','label','behavior_family','n_ev_in').with_columns(*[pl.Series(k,p) for k,p in pred.items()]);out.write_parquet(ROOT/'oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2));report=[]
 for (window,),g in out.group_by('window'):
  for kind in pred:report.append({'window':window,'kind':kind,'labelled_AP':average_precision_score(g['label'],g[kind]),'labelled_AP_negative_weight50':average_precision_score(g['label'],g[kind],sample_weight=np.where(g['label'].to_numpy()==1,1,50))})
 (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
