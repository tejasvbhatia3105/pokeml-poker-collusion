import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session8_data import hand_data,targets
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session13_action_residual');C=pl.col;torch.set_num_threads(2)
CONFIG={'steps':200,'learning_rate':.003,'bound':1.5,'residual_l2':.002,'weight_decay':.01,'loss':'priority-censored hand event Bernoulli likelihood via exact independent action noisy-OR','prior':'strict nested Cat head split equally across actions; at zero correction reproduces teacher exactly','arms':'linear and 32-unit nonlinear action correction; fixed schedule, train-only normalization'}
class Model(nn.Module):
 def __init__(self,n,kind):
  super().__init__();self.features=nn.Identity() if kind=='linear' else nn.Sequential(nn.Linear(n,32),nn.GELU());self.head=nn.Linear(n if kind=='linear' else 32,2);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x):return 1.5*torch.tanh(self.head(self.features(x)))
def bag_log_not(logits,g,n):return torch.zeros((n,2),dtype=logits.dtype).index_add(0,g,-nn.functional.softplus(logits))
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet');cols=[c for c in a.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family','time']];a=a.select('pair_id','hand_id',*cols).join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='m:1',maintain_order='left');X=a.select(cols).to_numpy().astype(np.float32);g=a['row'].to_numpy();cnt=np.bincount(g,minlength=len(d));assert cnt.min()>0;fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();pred={k:np.zeros((len(d),2)) for k in ['linear','nonlinear']};audit=[];start=time.time();identity=0.
 for f in range(4):
  r=d.select('pair_id','hand_id').join(pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');hp=r.select('cat_primary','cat_secondary').to_numpy().clip(1e-7,1-1e-7);pa=-np.expm1(np.log1p(-hp[g])/cnt[g,None]);prior=np.log(pa)-np.log1p(-pa);exact=-np.expm1(bag_log_not(torch.tensor(prior),torch.tensor(g.astype(np.int64)),len(d)).numpy());identity=max(identity,float(abs(exact-hp).max()))
  for family in ['directed_transfer','soft_play','coordinated_isolation']:
   y1,y2,e1,e2,va=targets(d,f,family);e=np.column_stack([e1,e2]);y=np.column_stack([y1,y2]).astype(np.float32);tr=np.flatnonzero((fv[g]!=f)&(fam[g]==family));vi=np.flatnonzero(va[g]);mu=X[tr].mean(0);sd=np.maximum(.05,X[tr].std(0));xx=torch.tensor(np.clip((X[tr]-mu)/sd,-6,6));vv=torch.tensor(np.clip((X[vi]-mu)/sd,-6,6));gi=torch.tensor(g[tr].astype(np.int64));gv=torch.tensor(g[vi].astype(np.int64));p0=torch.tensor(prior[tr],dtype=torch.float32);pv=torch.tensor(prior[vi],dtype=torch.float32);yy=torch.tensor(y);ee=torch.tensor(e);assert not np.any(va[g[tr]])
   for kind in pred:
    torch.manual_seed(13140+f);m=Model(len(cols),kind);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.01);trace=[]
    for step in range(200):
     delta=m(xx);ln=bag_log_not(p0+delta,gi,len(d));ev=(-torch.expm1(ln)).clamp_min(1e-12);loss=(-(yy*ev.log()+(1-yy)*ln))[ee].mean()+.002*delta.square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();trace.append(float(loss.detach()))
    root=ROOT/kind;root.mkdir(exist_ok=True);torch.save({'state_dict':m.state_dict(),'mu':mu,'sd':sd,'columns':cols,'kind':kind,'fold':f,'family':family,'config':CONFIG},root/f'{family}_fold{f}.pt');m.eval()
    with torch.no_grad():out=-torch.expm1(bag_log_not(pv+m(vv),gv,len(d))).numpy();pred[kind][va]=out[va]
    audit.append({'fold':f,'family':family,'kind':kind,'training_actions':len(tr),'train_validation_overlap':int(va[g[tr]].sum()),'loss_trace':trace})
   print('action residual',f,family,round(time.time()-start,1),flush=True)
 assert identity<1e-12;(ROOT/'audit.json').write_text(json.dumps({'identity_max_error':identity,'fits':audit},indent=2))
 for kind,pp in pred.items():
  root=ROOT/kind;d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pp[:,0]),pl.Series('bg_secondary',pp[:,1])).write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
