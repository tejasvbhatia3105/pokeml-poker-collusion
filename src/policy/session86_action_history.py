\
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
from catboost import CatBoostClassifier
ROOT=Path('artifacts/evidence_session86_action_history');C=pl.col;LENGTH=16
CONFIG={'history_length':16,'epochs':12,'batch_size':4096,'learning_rate':.001,'weight_decay':.001,'correction_penalty':.002,'correction_bound':3.,'seed':8600,'history_hidden':32,'arms':['summary','history'],'fit':'no early stopping; fixed schedule, no evidence labels','reference':'84 models exclude outer and training-row fold; validation mean of three outer-excluded references'}
PUBLIC_NAMES=[f'action_{i}' for i in range(4)]+[f'street_{i}' for i in range(4)]+['log_amount_bb','log_pot_bb','log_call_bb','log_stack_bb','players_active','relative_position','action_gap']+[f'{p}_{k}' for p in ['style','local_style'] for k in range(4)]+['current_actor','current_aggressor','same_street']
def history(a,indices):
 \
 n=len(a);ids=a['hand_id'].to_numpy();players=a['player_id'].to_numpy();starts=np.maximum.accumulate(np.where(np.r_[True,ids[1:]!=ids[:-1]],np.arange(n),0));lengths=np.minimum(indices-starts[indices],LENGTH);ix=indices[:,None]-lengths[:,None]+np.arange(LENGTH);valid=np.arange(LENGTH)[None,:]<lengths[:,None];ix=ix.clip(0,n-1);assert np.all(ids[ix][valid]==np.broadcast_to(ids[indices,None],ix.shape)[valid]);an=a['action_no'].to_numpy();assert np.all(an[ix][valid]<np.broadcast_to(an[indices,None],ix.shape)[valid]);street=a['street_no'].to_numpy().astype(int);action=a['action_class'].to_numpy();base=np.column_stack([np.eye(4,dtype=np.float32)[action],np.eye(4,dtype=np.float32)[street],a['log_amount_bb'].to_numpy(),np.log1p(a.select('pot_bb','call_bb','stack_bb').to_numpy().clip(0)),a['players_active'].to_numpy()/6,a['position'].to_numpy()/5,an/16,a.select([f'{p}_{k}' for p in ['style','local_style'] for k in range(4)]).to_numpy()]);out=np.zeros((len(indices),LENGTH,len(PUBLIC_NAMES)),np.float32);out[:,:,:23]=base[ix];out[:,:,13]=((a['position'].to_numpy()[ix]-a['position'].to_numpy()[indices,None])%6)/5;out[:,:,14]=(an[indices,None]-an[ix])/16;out[:,:,23]=(players[ix]==players[indices,None]);out[:,:,24]=(players[ix]==a['last_aggressor'].fill_null('').to_numpy()[indices,None]);out[:,:,25]=(street[ix]==street[indices,None]);out[~valid]=0;return out.astype(np.float16),lengths.astype(np.int64)
def prepare():
 ROOT.mkdir(exist_ok=True);pc=json.load(open('artifacts/policy/feature_columns.json'));tf=json.load(open('artifacts/policy/table_folds.json'));xs=[];hs=[];ls=[];ys=[];fs=[];meta=[];start=time.time();paths=sorted(Path('artifacts/policy/actions').glob('*.parquet'))
 for t,path in enumerate(paths):
  a=pl.read_parquet(path).filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');q=pl.concat([z.sample(n=min(n,len(z)),seed=414) for street,n in enumerate([600,300,150,150]) if len(z:=a.filter(C('street_no')==street))]);h,l=history(a,q['source_row'].to_numpy());xs.append(q.select(pc).to_numpy());hs.append(h);ls.append(l);ys.append(q['action_class'].to_numpy());fs.append(np.full(len(q),tf[path.stem],np.int8));meta.append(q.select('table_id','hand_id','player_id','action_no','source_row'));assert not (((q['call_bb'].to_numpy()<=0)&np.isin(ys[-1],[0,2]))|((q['call_bb'].to_numpy()>0)&(ys[-1]==1))).any()
  if (t+1)%50==0:print('history prepared tables',t+1,round(time.time()-start,1),flush=True)
 np.savez_compressed(ROOT/'sample.npz',x=np.concatenate(xs),history=np.concatenate(hs),length=np.concatenate(ls),y=np.concatenate(ys),fold=np.concatenate(fs));pl.concat(meta).write_parquet(ROOT/'sample_keys.parquet');(ROOT/'config.json').write_text(json.dumps({'method':__doc__,**CONFIG,'current_columns':pc,'history_columns':PUBLIC_NAMES,'sample_per_table_street':[600,300,150,150],'sample_seed':414},indent=2));print('prepared',sum(map(len,xs)),round(time.time()-start,1),flush=True)
def current_transform(x,pc):
 x=x.copy()
 for c in ['big_blind','pot_bb','call_bb','stack_bb']:x[:,pc.index(c)]=np.log1p(x[:,pc.index(c)].clip(0))
 return x.astype(np.float32)
def reference(x,native,f):
 from session84_nested_joint_policy import paths
 out=np.zeros((len(x),4),np.float64)
 for h in range(4):
  if h==f:continue
  m=CatBoostClassifier();m.load_model(str(paths(f,h)[0]));mask=(native==f)|(native==h);p=m.predict_proba(x[mask],thread_count=3);ix=np.flatnonzero(mask);out[ix[native[mask]==h]]=p[native[mask]==h];out[ix[native[mask]==f]]+=p[native[mask]==f]/3
 return np.log(out.clip(1e-7)).astype(np.float32)
class Policy(nn.Module):
 def __init__(self,kind,current=33):
  super().__init__();self.kind=kind
  if kind=='history':self.gru=nn.GRU(len(PUBLIC_NAMES),32,batch_first=True)
  self.net=nn.Sequential(nn.Linear(current+(32 if kind=='history' else 0),128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,4));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x,h,length):
  if self.kind=='history':
   z,_=self.gru(h);z=z[torch.arange(len(x)),(length-1).clamp_min(0)];z=z*(length>0)[:,None];x=torch.cat([x,z],1)
  return 3*torch.tanh(self.net(x)/3)
def legal_logits(z,call):
 z=z.clone();z[call,1]=-30;z[~call,0]=-30;z[~call,2]=-30;return z
def predict(m,x,h,l,prior,call,indices):
 chunks=[];m.eval()
 with torch.no_grad():
  for st in range(0,len(indices),CONFIG['batch_size']):
   ix=indices[st:st+CONFIG['batch_size']];delta=m(torch.from_numpy(x[ix]),torch.from_numpy(h[ix].astype(np.float32)),torch.from_numpy(l[ix]));logits=legal_logits(torch.from_numpy(prior[ix])+delta,torch.from_numpy(call[ix]));chunks.append(torch.softmax(logits,1).numpy())
 return np.concatenate(chunks)
def train(folds):
 torch.set_num_threads(3);z=np.load(ROOT/'sample.npz');raw=z['x'];h=z['history'];length=z['length'];y=z['y'].astype(np.int64);native=z['fold'];pc=json.load(open('artifacts/policy/feature_columns.json'));allx=current_transform(raw,pc);call=raw[:,pc.index('call_bb')]>0;start=time.time()
 for f in folds:
  tr=native!=f;va=~tr;mean=allx[tr].mean(0);scale=np.maximum(allx[tr].std(0),.01);x=((allx-mean)/scale).clip(-20,20).astype(np.float32);prior=reference(raw,native,f);np.savez_compressed(ROOT/f'inputs_fold{f}.npz',prior=prior,mean=mean,scale=scale);vi=np.flatnonzero(va);ti=np.flatnonzero(tr);baseline=torch.softmax(legal_logits(torch.from_numpy(prior[va]),torch.from_numpy(call[va])),1).numpy();np.save(ROOT/f'reference_fold{f}.npy',baseline);results={}
  for kind in CONFIG['arms']:
   torch.manual_seed(CONFIG['seed']+f);rng=np.random.default_rng(CONFIG['seed']+f);m=Policy(kind,len(pc));opt=torch.optim.AdamW(m.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CONFIG['epochs'],eta_min=.0001);trace=[]
   for epoch in range(CONFIG['epochs']):
    m.train();indices=rng.permutation(ti);total=0
    for st in range(0,len(indices),CONFIG['batch_size']):
     ix=indices[st:st+CONFIG['batch_size']];delta=m(torch.from_numpy(x[ix]),torch.from_numpy(h[ix].astype(np.float32)),torch.from_numpy(length[ix]));logits=legal_logits(torch.from_numpy(prior[ix])+delta,torch.from_numpy(call[ix]));loss=nn.functional.cross_entropy(logits,torch.from_numpy(y[ix]))+CONFIG['correction_penalty']*delta.square().mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();total+=float(loss.detach())*len(ix)
    sched.step();trace.append(total/len(ti));print('history fit',f,kind,epoch+1,trace[-1],round(time.time()-start,1),flush=True)
   m.eval();torch.save({'state':m.state_dict(),'kind':kind,'mean':mean,'scale':scale,'config':CONFIG,'fold':f},ROOT/f'{kind}_fold{f}.pt');p=predict(m,x,h,length,prior,call,vi);np.save(ROOT/f'{kind}_fold{f}.npy',p);results[kind]={'heldout_logloss':float(-np.log(p[np.arange(len(vi)),y[vi]].clip(1e-7)).mean()),'training_loss':trace}
  results['reference']={'heldout_logloss':float(-np.log(baseline[np.arange(len(vi)),y[vi]].clip(1e-7)).mean())};results['train_actions']=len(ti);results['valid_actions']=len(vi);(ROOT/f'metrics_fold{f}.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 else:train([int(f) for f in sys.argv[2].split(',')])
