\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('artifacts/evidence_session116_relationship_encoder')
DATA=Path('artifacts/evidence_session115_relationship_data');C=pl.col
CONFIG={'method':__doc__,'d':128,'layers':2,'heads':4,'dropout':.15,'epochs':10,'batch':32,
        'learning_rate':.0003,'weight_decay':.01,'gradient_clip':2.,'auxiliary_weight':.5,
        'unlisted_auxiliary_weight':.1,'family_weight':.3,'seed_base':11601,
        'loss':'mean trusted pair BCE + .3 positive family CE + .5 mean of local/context auxiliary BCE; auxiliary normalized per positive relationship',
        'role_swap':'random raw-space swap during training; both roles averaged at scoring',
        'normalization':'training-pool mean/std only, std floor1e-3; normalized features clipped[-10,10]',
        'validation':'fixed10 epochs, no held-out stopping; score excluded folds only; no truncation'}
FAMILIES=['none','directed_transfer','soft_play','coordinated_isolation']
torch.set_num_threads(2)

def load():
    cfg=json.loads((DATA/'config.json').read_text());x=np.load(DATA/'x.npy',mmap_mode='r')
    assert hashlib.file_digest((DATA/'x.npy').open('rb'),'sha256').hexdigest()==cfg['array_sha256']
    b=pl.read_parquet(DATA/'bags.parquet');h=pl.read_parquet(DATA/'hands.parquet')
    return cfg,x,b,h

def normalization(x,b,train):
    ix=np.concatenate([np.arange(r['offset'],r['offset']+r['n_hands']) for r in b[train].to_dicts()])
    z=np.asarray(x[ix],np.float64);mu=z.mean(0).astype(np.float32);sd=np.maximum(z.std(0),1e-3).astype(np.float32)
    return mu,sd,ix

def batch(x,b,h,indices,mu,sd,cfg,swap=False):
    records=b[indices].to_dicts();length=max(r['n_hands'] for r in records)
    xx=np.zeros((len(records),length,x.shape[1]),np.float32);mask=np.zeros((len(records),length),bool);ev=np.zeros_like(mask,dtype=np.float32)
    lab=np.array([r['label'] for r in records],np.float32);family=np.array([FAMILIES.index(r['behavior_family']) for r in records],np.int64)
    y=h['evidence'].to_numpy();perm=cfg['role_permutation'];seat=cfg['seat_distance_column']
    for i,r in enumerate(records):
        start,n=r['offset'],r['n_hands'];raw=np.array(x[start:start+n],copy=True)
        if swap:raw=raw[:,perm];raw[:,seat]=6-raw[:,seat]
        xx[i,:n]=np.clip((raw-mu)/sd,-10,10);mask[i,:n]=True;ev[i,:n]=y[start:start+n]
    return tuple(torch.from_numpy(z) for z in [xx,mask,ev,lab,family])

class Model(nn.Module):
    def __init__(self,nf,time_col):
        super().__init__();d=CONFIG['d'];self.time_col=time_col
        self.input=nn.Sequential(nn.Linear(nf,d),nn.GELU(),nn.LayerNorm(d),nn.Dropout(CONFIG['dropout']))
        self.time=nn.Linear(1,d)
        layer=nn.TransformerEncoderLayer(d,CONFIG['heads'],dim_feedforward=2*d,dropout=CONFIG['dropout'],batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,CONFIG['layers'],enable_nested_tensor=False)
        self.attention=nn.Linear(d,1);self.local=nn.Linear(d,1);self.context=nn.Linear(d,1)
        self.pair=nn.Sequential(nn.Linear(3*d+1,d),nn.GELU(),nn.Dropout(CONFIG['dropout']),nn.Linear(d,4))
    def forward(self,x,mask):
        local=self.input(x)+self.time(x[:,:,self.time_col,None]);context=self.encoder(local,src_key_padding_mask=~mask)
        weight=self.attention(context).squeeze(-1).masked_fill(~mask,-1e4).softmax(1)
        mean=(context*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True)
        maximum=context.masked_fill(~mask[:,:,None],-1e4).max(1).values
        pooled=(weight[:,:,None]*context).sum(1)
        pair=self.pair(torch.cat([pooled,mean,maximum,mask.sum(1,keepdim=True).float().log1p()],1))
        return pair,self.local(local).squeeze(-1),self.context(context).squeeze(-1)

def objective(outputs,mask,ev,label,family):
    pair,local,context=outputs;loss=F.binary_cross_entropy_with_logits(pair[:,0],label);positive=label>0
    if positive.any():
        loss=loss+CONFIG['family_weight']*F.cross_entropy(pair[positive,1:],family[positive]-1)
        weight=torch.where(ev>0,1.,CONFIG['unlisted_auxiliary_weight'])*mask
        auxiliary=0.
        for logits in [local,context]:
            per=(F.binary_cross_entropy_with_logits(logits,ev,reduction='none')*weight).sum(1)/weight.sum(1).clamp(min=1)
            auxiliary=auxiliary+per[positive].mean()/2
        loss=loss+CONFIG['auxiliary_weight']*auxiliary
    return loss

def predict(model,x,b,h,indices,mu,sd,cfg,device):
    model.eval();parts=[]
    with torch.no_grad():
        for pos in range(0,len(indices),16):
            selected=indices[pos:pos+16];probability=[]
            for swap in [False,True]:
                xx,mask,*_=batch(x,b,h,selected,mu,sd,cfg,swap);pair,local,context=model(xx.to(device),mask.to(device))
                probability.append((torch.sigmoid(local).cpu().numpy(),torch.sigmoid(context).cpu().numpy(),torch.sigmoid(pair[:,0]).cpu().numpy(),torch.softmax(pair[:,1:],1).cpu().numpy()))
            probs=[(a+z)/2 for a,z in zip(*probability)]
            for i,r in enumerate(b[selected].to_dicts()):
                start,n=r['offset'],r['n_hands'];z=h.slice(start,n).select('row','pair_id','hand_id')
                z=z.with_columns(pl.Series('local',probs[0][i,:n]),pl.Series('context',probs[1][i,:n]),pl.lit(float(probs[2][i])).alias('pair_risk'),
                    *[pl.lit(float(probs[3][i,j])).alias('family_'+str(j)) for j in range(3)])
                parts.append(z)
    return pl.concat(parts).sort('row')

def train():
    ROOT.mkdir(exist_ok=True);cfg,x,b,h=load();device=torch.device('mps')
    assert torch.backends.mps.is_available(),'Run the authorized MPS training outside the sandbox; do not silently start a long CPU job.'
    print('DEVICE',device,flush=True);plans=json.loads((DATA/'reference_plan.json').read_text());start=time.time()
    manifest=dict(config=CONFIG,data_config=cfg,source_sha256=hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest())
    path=ROOT/'config.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:path.write_text(json.dumps(manifest,indent=2))
    for plan in plans:
        f,g=plan['excluded_folds'];name=f'exclude{f}{g}';checkpoint=ROOT/(name+'.pt');output=ROOT/(name+'.parquet');auditpath=ROOT/(name+'.json')
        if checkpoint.exists() and output.exists() and auditpath.exists():
            audit=json.loads(auditpath.read_text());assert audit['manifest']==manifest and audit['reference_plan']==plan
            print('RESUME_COMPLETE',name,flush=True);continue
        tr=np.flatnonzero(~b['fold'].is_in([f,g]).to_numpy());va=np.flatnonzero(b['fold'].is_in([f,g]).to_numpy())
        assert set(b[tr]['pair_id'])==set(plan['training_pair_ids'])
        mu,sd,rows=normalization(x,b,tr);seed=CONFIG['seed_base']+4*f+g;torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=Model(x.shape[1],cfg['columns'].index('trel')).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
        batches=(len(tr)+CONFIG['batch']-1)//CONFIG['batch'];scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=CONFIG['learning_rate'],total_steps=CONFIG['epochs']*batches,pct_start=.1)
        losses=[]
        for epoch in range(CONFIG['epochs']):
            order=rng.permutation(tr);model.train();total=0.
            for pos in range(0,len(order),CONFIG['batch']):
                data=batch(x,b,h,order[pos:pos+CONFIG['batch']],mu,sd,cfg,bool(rng.integers(2)))
                xx,mask,ev,label,family=[t.to(device) for t in data];optimizer.zero_grad(set_to_none=True)
                loss=objective(model(xx,mask),mask,ev,label,family);assert torch.isfinite(loss)
                loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip']);assert torch.isfinite(norm)
                optimizer.step();scheduler.step();total+=float(loss.detach().cpu())
            losses.append(total/batches);print('RELATIONSHIP_EPOCH',name,epoch+1,losses[-1],time.time()-start,flush=True)
        result=predict(model,x,b,h,va,mu,sd,cfg,device);assert len(result)==int(b[va]['n_hands'].sum())
        result.write_parquet(output)
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mu=mu,sd=sd,config=CONFIG,data_config=cfg,excluded_folds=[f,g]),checkpoint)
        audit=dict(manifest=manifest,reference_plan=plan,training_pairs=len(tr),scored_pairs=len(va),training_rows=len(rows),scored_rows=len(result),losses=losses,
            checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),seconds=time.time()-start)
        auditpath.write_text(json.dumps(audit,indent=2));print('RELATIONSHIP_REFERENCE_DONE',name,time.time()-start,flush=True)
        del model,optimizer;torch.mps.empty_cache()

def smoke():
    cfg,x,b,h=load();tr=np.flatnonzero(b['fold'].to_numpy()<2);mu,sd,_=normalization(x,b,tr)
    selected=tr[:4];xx,mask,ev,label,family=batch(x,b,h,selected,mu,sd,cfg)
    torch.manual_seed(11601);model=Model(x.shape[1],cfg['columns'].index('trel'));loss=objective(model(xx,mask),mask,ev,label,family)
    loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert int(mask.sum())==int(b[selected]['n_hands'].sum())
    model.eval()
    with torch.no_grad():
        first=model(xx,mask);changed=xx.clone();changed[:,1:]*=-1;second=model(changed,mask)
        np.testing.assert_array_equal(first[1][:,0].numpy(),second[1][:,0].numpy())
        assert float(abs(first[2][:,0]-second[2][:,0]).max())>1e-6
    ROOT.mkdir(exist_ok=True);report=dict(loss=float(loss.detach()),hand_rows=int(mask.sum()),local_query_context_mutation_error=0,
        context_query_mutation_difference=float(abs(first[2][:,0]-second[2][:,0]).max()),finite_gradients=True)
    (ROOT/'smoke.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':{'smoke':smoke,'train':train}[sys.argv[1]]()
