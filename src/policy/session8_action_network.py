\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session8_data import hand_data,targets,ROOT,C
from session6_priority import inclusion
torch.set_num_threads(4)
class Net(nn.Module):
    def __init__(self,na,nh):
        super().__init__();self.action=nn.Sequential(nn.Linear(na,64),nn.GELU(),nn.Linear(64,64),nn.GELU());self.hand=nn.Sequential(nn.Linear(nh,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,64),nn.GELU());self.out=nn.Sequential(nn.Linear(195,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,2))
    def forward(self,a,h,mask,family):
        z=self.action(a);mean=(z*mask[:,:,None]).sum(1)/mask.sum(1).clamp(min=1)[:,None];maximum=z.masked_fill(~mask[:,:,None],-1e5).max(1).values;return self.out(torch.cat([mean,maximum,self.hand(h),nn.functional.one_hot(family,3).float()],1))
def main():
    d=hand_data();a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet');ac=[c for c in a.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family','time']];a=a.select('pair_id','hand_id',*ac).join(pl.read_parquet(ROOT/'neighbor_actions.parquet'),on=['pair_id','hand_id','action_no'],validate='1:1').join(d.select('pair_id','hand_id',C('row').alias('hand_row')),on=['pair_id','hand_id'],validate='m:1').sort('hand_row','action_no');ac=[c for c in a.columns if c not in ['pair_id','hand_id','hand_row']];hc=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['event'];AX=a.select(ac).to_numpy().astype('float32');HX=d.select(hc).to_numpy().astype('float32');group=a['hand_row'].to_numpy();counts=np.bincount(group,minlength=len(d));assert counts.min()>0;slots=np.zeros((len(d),counts.max()),np.int64);mask=np.arange(counts.max())[None,:]<counts[:,None];offset=0
    for i,n in enumerate(counts):slots[i,:n]=np.arange(offset,offset+n);offset+=n
    families=['directed_transfer','soft_play','coordinated_isolation'];family=np.array([families.index(b) for b in d['behavior_family']]);fold=d['fold'].to_numpy();pred=np.zeros((len(d),2));start=time.time();logs=[]
    for f in range(4):
        torch.manual_seed(8100+f);rng=np.random.default_rng(8100+f);tr=fold!=f;va=~tr;Y=np.zeros((len(d),2),np.float32);E=np.zeros_like(Y)
        for b in families:
            t1,t2,e1,e2,_=targets(d,f,b);Y[:,0]+=t1;Y[:,1]+=t2;E[:,0]+=e1;E[:,1]+=e2
        am=AX[tr[group]].mean(0);ast=AX[tr[group]].std(0).clip(.001);hm=HX[tr].mean(0);hst=HX[tr].std(0).clip(.001);AT=torch.tensor(np.clip((AX-am)/ast,-10,10));HT=torch.tensor(np.clip((HX-hm)/hst,-10,10));YT=torch.tensor(Y);ET=torch.tensor(E);FT=torch.tensor(family);MT=torch.tensor(mask);ST=torch.tensor(slots)
        model=Net(AX.shape[1],HX.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);indices=np.flatnonzero(tr)
        def batch(ix):return AT[ST[ix]],HT[ix],MT[ix],FT[ix]
        for epoch in range(25):
            rng.shuffle(indices);model.train();losses=[]
            for beg in range(0,len(indices),256):
                ix=indices[beg:beg+256];opt.zero_grad();v=model(*batch(ix));loss=(nn.functional.binary_cross_entropy_with_logits(v,YT[ix],reduction='none')*ET[ix]).sum()/ET[ix].sum().clamp(min=1);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
            if epoch in [0,9,24]:print('action network fold',f,'epoch',epoch+1,'loss',round(np.mean(losses),4),'seconds',round(time.time()-start,1),flush=True)
        model.eval()
        with torch.no_grad():
            indices=np.flatnonzero(va)
            for beg in range(0,len(indices),512):ix=indices[beg:beg+512];pred[ix]=torch.sigmoid(model(*batch(ix))).numpy()
        torch.save({'state':model.state_dict(),'action_columns':ac,'hand_columns':hc,'action_mean':am,'action_std':ast,'hand_mean':hm,'hand_std':hst,'epochs':25,'fold':f},ROOT/f'action_network_fold{f}.pt')
    q=d.select('pair_id','hand_id','time').with_columns(pl.Series('primary',pred[:,0]),pl.Series('secondary',pred[:,1]));parts=[]
    for _,g in q.group_by('pair_id',maintain_order=True):parts.append(g.with_columns(pl.Series('score',inclusion(g['primary'].to_numpy(),g['secondary'].to_numpy()))))
    pl.concat(parts).write_parquet(ROOT/'action_network_oof.parquet')
if __name__=='__main__':main()
