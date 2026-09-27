\
\
\
\
import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');sys.path.insert(0,'src/policy')
from pathlib import Path
import json,time
import numpy as np,polars as pl
import torch
from torch import nn
from scipy.special import logit
from session4_set_evidence import design,ap
torch.set_num_threads(4)
OUT=Path('artifacts/evidence_session4')
class Net(nn.Module):
    def __init__(self,n):
        super().__init__();self.enc=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,32))
        self.family=nn.Embedding(3,32)
        layer=nn.TransformerEncoderLayer(32,4,64,.2,activation='gelu',batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.out=nn.Linear(32,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,f,base):
        z=self.enc(x)+self.family(f)[:,None,:];z=self.context(z)
        return base+2*torch.tanh(self.out(z).squeeze(-1))
def main():
    d=pl.read_parquet('artifacts/policy/evidence_training.parquet').join(pl.read_parquet(OUT/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
    raw=pl.read_parquet(OUT/'raw_action_features.parquet');rawcols=[c for c in raw.columns if c.startswith('raw_action_')]
    d=d.join(raw,on=['pair_id','hand_id']);families=['directed_transfer','soft_play','coordinated_isolation'];rows=[];t=time.time()
    for f in range(4):
        torch.manual_seed(4410+f);rng=np.random.default_rng(4410+f)
        q=d.join(pl.read_parquet(OUT/f'nested_outer{f}.parquet'),on=['pair_id','hand_id']);bags=[];xx=[]
        for _,g in q.group_by('pair_id',maintain_order=True):
            b=design(g);bags.append(b);z=pl.DataFrame({'hand_id':b['hand']}).join(g,on='hand_id',maintain_order='left')
                                                                       
            xx.append(np.concatenate([z.select(rawcols).to_numpy(),b['U']],1))
        x=np.stack(xx).astype('float32');y=np.stack([b['y'] for b in bags]).astype('float32')
        base=np.stack([b['U'][:,0] for b in bags]);fam=np.array([families.index(b['family']) for b in bags]);va=np.array([b['fold']==f for b in bags]);tr=~va
        flat=x[tr].reshape(-1,x.shape[-1]);mu=np.median(flat,axis=0);sd=np.quantile(flat,.9,axis=0)-np.quantile(flat,.1,axis=0);sd=np.maximum(sd,.1)
        x=np.clip((x-mu)/sd,-5,5).astype('float32');X=torch.tensor(x);Y=torch.tensor(y);B=torch.tensor(base);F=torch.tensor(fam)
        net=Net(x.shape[-1]);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.03)
        ids=np.flatnonzero(tr);history=[]
        for epoch in range(40):
            net.train();order=rng.permutation(ids);ls=[]
            for start in range(0,len(order),32):
                ix=order[start:start+32];out=net(X[ix],F[ix],B[ix]);yy=Y[ix]
                bce=nn.functional.binary_cross_entropy_with_logits(out,yy)
                target=yy/yy.sum(1,keepdim=True).clamp(min=1);ll=-(target*nn.functional.log_softmax(out,1)).sum(1).mean()
                loss=bce+.5*ll;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();ls.append(float(loss.detach()))
            history.append(float(np.mean(ls)))
        net.eval()
        with torch.no_grad():pred=net(X[va],F[va],B[va]).numpy()
                                                                                  
        for b,s in zip([b for b in bags if b['fold']==f],pred):
            rows.append(dict(pair_id=b['pid'],fold=f,family=b['family'],base=ap(b,list(range(5))),neural=ap(b,np.argsort(-s,kind='stable')[:5])))
                                                                         
        perm=torch.arange(11,-1,-1)
        with torch.no_grad():
            ref=net(X[va][:2],F[va][:2],B[va][:2]);alt=net(X[va][:2,perm],F[va][:2],B[va][:2,perm])[:,perm]
            assert torch.allclose(ref,alt,atol=2e-5)
        torch.save(net.state_dict(),OUT/f'neural_set_fold{f}.pt');np.savez(OUT/f'neural_set_norm{f}.npz',mu=mu,sd=sd)
        (OUT/f'neural_history{f}.json').write_text(json.dumps(history))
        print('neural fold',f,'seconds',round(time.time()-t,1),flush=True)
    r=pl.DataFrame(rows);r.write_csv(OUT/'neural_set_validation.csv');print(r.group_by('family').agg(pl.col('base','neural').mean()))
    print('overall',r.select(pl.col('base','neural').mean()).to_dicts(),flush=True)
if __name__=='__main__':main()
