\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score as ap

root=Path('artifacts/policy/joint_bags')
parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,default=0);parser.add_argument('--epochs',type=int,default=40);parser.add_argument('--withheld-family',choices=['directed_transfer','soft_play','coordinated_isolation']);args=parser.parse_args()
torch.set_num_threads(4);torch.manual_seed(811+args.fold);rng=np.random.default_rng(811+args.fold)
bags=pd.read_csv(root/'bags.csv');raw=np.load(root/'features.npy');ev=np.load(root/'evidence.npy');folds=json.loads(Path('artifacts/folds.json').read_text());va=bags.table_id.isin(folds[args.fold]['valid_tables']).to_numpy();tr=~va
withheld=['none','directed_transfer','soft_play','coordinated_isolation'].index(args.withheld_family) if args.withheld_family else -1
if args.withheld_family:
    tr &= bags.behavior.to_numpy()!=withheld
    root=root/('withheld_'+args.withheld_family);root.mkdir(exist_ok=True)
rowtrain=np.zeros(len(raw),bool)
for row in bags[tr].itertuples():rowtrain[row.start:row.end]=True
sample=rng.choice(np.flatnonzero(rowtrain),size=min(50000,int(rowtrain.sum())),replace=False)
median=np.median(raw[sample],axis=0);scale=np.maximum(np.quantile(raw[sample],.9,axis=0)-np.quantile(raw[sample],.1,axis=0),.01)
z=(raw-median)/scale;x=(np.sign(z)*np.log1p(abs(z))).clip(-8,8).astype(np.float32);del raw,z
np.savez(root/f'scaler_fold{args.fold}.npz',median=median,scale=scale)

class JointBags(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(n,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,32),nn.SiLU())
        self.attention=nn.Sequential(nn.Linear(96,32),nn.SiLU(),nn.Linear(32,1))
        self.head=nn.Sequential(nn.Linear(96,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,4))
    def forward(self,x,mask):
        h=self.encode(x);mean=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        maximum=h.masked_fill(~mask[:,:,None],-1e4).amax(1);context=torch.cat([mean,maximum],1)
        score=self.attention(torch.cat([h,context[:,None,:].expand(-1,h.shape[1],-1)],2)).squeeze(-1).masked_fill(~mask,-1e4)
        weights=torch.softmax(score,1);selected=(h*weights[:,:,None]).sum(1)
        output=self.head(torch.cat([context,selected],1))
        return output[:,0],output[:,1:],score

def batch(indices):
    rows=bags.iloc[indices];lengths=(rows.end-rows.start).to_numpy();n=int(lengths.max());xx=np.zeros((len(rows),n,x.shape[1]),np.float32);ee=np.zeros((len(rows),n),np.float32);mask=np.zeros((len(rows),n),bool)
    for i,row in enumerate(rows.itertuples()):
        length=row.end-row.start;xx[i,:length]=x[row.start:row.end];ee[i,:length]=ev[row.start:row.end];mask[i,:length]=True
    return torch.from_numpy(xx),torch.from_numpy(mask),torch.from_numpy(ee),torch.tensor(rows.label.to_numpy(),dtype=torch.float32),torch.tensor(rows.behavior.to_numpy()-1,dtype=torch.long)

model=JointBags(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);t=time.time();history=[]
def evaluate():
    model.eval();pred=[];evidence=[];scores=[];idx=np.flatnonzero(va)
    with torch.no_grad():
        for start in range(0,len(idx),32):
            take=idx[start:start+32];xx,mask,ee,y,behavior=batch(take);risk,family,score=model(xx,mask);risk=torch.sigmoid(risk).numpy();family=family.softmax(1).numpy();score=score.numpy()
            for j,bagi in enumerate(take):
                row=bags.iloc[bagi];length=int(row.end-row.start);pred.append(dict(pair_id=row.pair_id,risk=float(risk[j]),truth=int(row.label),truth_behavior=int(row.behavior),family=int(family[j].argmax()+1)))
                scores.append((int(row.start),int(row.end),score[j,:length]))
                if row.label:
                    order=np.argsort(-score[j,:length],kind='stable')[:5];rel=ev[row.start:row.end][order];emap=float((np.cumsum(rel)*rel/np.arange(1,len(rel)+1)).sum()/min(5,ev[row.start:row.end].sum()));evidence.append(dict(pair_id=row.pair_id,map5=emap,truth_behavior=int(row.behavior)))
    p=pd.DataFrame(pred);y=p.truth.to_numpy();risk=p.risk.to_numpy();report=dict(epoch=epoch,pair_AP=ap(y,risk),weighted_pair_AP=ap(y,risk,sample_weight=np.where(y>0,1,50)),evidence_MAP5=float(np.mean([e['map5'] for e in evidence])),seconds=round(time.time()-t,1))
    if args.withheld_family:
        keep=(y==0)|(p.truth_behavior.to_numpy()==withheld);report['withheld_weighted_AP']=ap(y[keep],risk[keep],sample_weight=np.where(y[keep]>0,1,50));report['withheld_evidence_MAP5']=float(np.mean([e['map5'] for e in evidence if e['truth_behavior']==withheld]))
    return report,p,pd.DataFrame(evidence),scores

for epoch in range(1,args.epochs+1):
    model.train();indices=rng.permutation(np.flatnonzero(tr));total=0
    for start in range(0,len(indices),32):
        xx,mask,ee,y,behavior=batch(indices[start:start+32]);risk,family,score=model(xx,mask)
        loss=nn.functional.binary_cross_entropy_with_logits(risk,y)
        positive=y>0
        if positive.any():
            targets=ee[positive]/ee[positive].sum(1,keepdim=True)
            ranking=-(targets*nn.functional.log_softmax(score[positive],1)).sum(1).mean()
            loss=loss+.5*ranking+.2*nn.functional.cross_entropy(family[positive],behavior[positive])
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=float(loss.detach())
    if epoch%10==0 or epoch==args.epochs:
        report,p,e,scores=evaluate();history.append(report);print('JOINT',report,flush=True)
        (root/f'history_fold{args.fold}.json').write_text(json.dumps(history,indent=2))
torch.save(model.state_dict(),root/f'model_fold{args.fold}.pt');p.to_csv(root/f'pair_oof_fold{args.fold}.csv',index=False);e.to_csv(root/f'evidence_oof_fold{args.fold}.csv',index=False)
allscore=np.full(len(x),np.nan,np.float32)
for start,end,score in scores:allscore[start:end]=score
np.save(root/f'hand_scores_fold{args.fold}.npy',allscore)
