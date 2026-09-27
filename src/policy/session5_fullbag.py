\
\
\
\
\
import os, ast, json, time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np, polars as pl, torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(4)
OUT=Path('artifacts/evidence_session5'); MD=Path('artifacts/seq_v6s1')
cfg=json.loads((MD/'config.json').read_text())
local=json.loads(Path('artifacts/seq_v6/config.json').read_text())
T2=Path(local['tokens2']);T3=Path(local['tokens3']);T=T2.parent/'seq_tokens'
cols=json.loads((T/'columns.json').read_text())+json.loads((T2/'columns2.json').read_text())+json.loads((T3/'columns3.json').read_text())
ns=dict(cfg=cfg,nn=nn,torch=torch,ACT=False,F_IN=len(cols),FS=0,TREL=cols.index('trel'))
tree=ast.parse(Path('src/policy/seq_train.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Net','swapname']],type_ignores=[]),'<sequence definitions>','exec'),ns)
perm=np.array([cols.index(ns['swapname'](c)) for c in cols]);seatd=cols.index('seatd')
nz=np.load(MD/'norm.npz');mu=nz['mu'];sd=nz['sd']
ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet')
families=['directed_transfer','soft_play','coordinated_isolation'];bags=[]
for (table,),q in ix.group_by('table_id',maintain_order=True):
    with np.load(T/f'{table}.npz') as z:
        offsets=np.r_[0,np.cumsum(z['n'])]
        parts=[np.load(p,mmap_mode='r') for p in [T/f'{table}_X.npy',T2/f'{table}.npy',T3/f'{table}.npy']]
        pm={p:i for i,p in enumerate(z['pair_id']) if z['phase'][i]=='development'}
        for (pid,),g in q.group_by('pair_id',maintain_order=True):
            a,b=offsets[pm[pid]:pm[pid]+2];hids=z['hand_id'][a:b]
            truth=set(g.filter(pl.col('evidence')==1)['hand_id'])
            x=(np.concatenate([v[a:b] for v in parts],1).astype('float32')-mu)/sd
            bags.append(dict(pair_id=pid,hand_id=hids,x=x,y=np.array([h in truth for h in hids],np.float32),fold=g['fold'][0],family=families.index(g['behavior_family'][0])))
assert sum(b['y'].sum() for b in bags)==1817

def batch(bs,swap=False):
    n=max(len(b['x']) for b in bs);x=np.zeros((len(bs),n,len(cols)),np.float32);y=np.zeros((len(bs),n),np.float32);m=np.zeros_like(y,bool)
    for i,b in enumerate(bs):
        k=len(b['x']);x[i,:k]=b['x'];y[i,:k]=b['y'];m[i,:k]=True
    if swap:
        x=x[:,:,perm].copy();x[:,:,seatd]=(6-x[:,:,seatd]*sd[seatd]-mu[seatd])/sd[seatd]
    return torch.tensor(x),torch.tensor(m),torch.tensor(y),torch.tensor([b['family'] for b in bs])

def logits(model,head,x,m,fam):
    h=model.enc(model.inp(x)+model.pos(x[:,:,cols.index('trel'),None]),src_key_padding_mask=~m)
    return head(h).gather(2,fam[:,None,None].expand(-1,x.shape[1],1)).squeeze(-1)

results=[];start=time.time()
for fold in range(4):
    torch.manual_seed(510+fold);rng=np.random.default_rng(510+fold)
    model=ns['Net'](len(cols));model.load_state_dict(torch.load(MD/f'seq_fold{fold}.pt',map_location='cpu',weights_only=True),strict=True)
    head=nn.Linear(model.att.in_features,3);nn.init.constant_(head.bias,-3.)
    opt=torch.optim.AdamW([{'params':model.parameters(),'lr':3e-5},{'params':head.parameters(),'lr':1e-3}],weight_decay=.02)
    train=[b for b in bags if b['fold']!=fold];valid=[b for b in bags if b['fold']==fold]
    for epoch in range(25):
        model.train();head.train();order=rng.permutation(len(train));losses=[]
        for k in range(0,len(order),16):
            bs=[train[i] for i in order[k:k+16]];x,m,y,fam=batch(bs,swap=bool(rng.integers(2)))
            z=logits(model,head,x,m,fam)
            bce=F.binary_cross_entropy_with_logits(z,y,reduction='none',pos_weight=torch.tensor(8.))
            bce=((bce*m).sum(1)/m.sum(1)).mean()
            lse=F.log_softmax(z.masked_fill(~m,-1e4),1)
            rank=(-(lse*y).sum(1)/y.sum(1)).mean()
            loss=bce+.3*rank;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.);opt.step();losses.append(float(loss.detach()))
        if epoch in [9,24]:
            model.eval();head.eval();rr=[]
            with torch.no_grad():
                for k in range(0,len(valid),16):
                    bs=valid[k:k+16];pred=0
                    for sw in [False,True]:
                        x,m,y,fam=batch(bs,sw);pred=pred+torch.sigmoid(logits(model,head,x,m,fam)).numpy()/2
                    for i,b in enumerate(bs):rr.append(pl.DataFrame({'pair_id':[b['pair_id']]*len(b['x']),'hand_id':b['hand_id'],'score':pred[i,:len(b['x'])]}))
            r=pl.concat(rr);r.write_parquet(OUT/f'fullbag_epoch{epoch+1}_fold{fold}.parquet')
            if epoch==24:
                results.append(r);torch.save({'encoder':model.state_dict(),'evidence_head':head.state_dict()},OUT/f'fullbag_fold{fold}.pt')
        if epoch%5==4:print('fold',fold,'epoch',epoch+1,'loss',round(np.mean(losses),4),'seconds',round(time.time()-start,1),flush=True)
pl.concat(results).write_parquet(OUT/'fullbag_oof.parquet')
(OUT/'fullbag_config.json').write_text(json.dumps({'pretrained':str(MD),'epochs':25,'encoder_lr':3e-5,'head_lr':1e-3,'positive_weight':8,'listwise_weight':.3,'columns':cols,'label_version':'pair_hand_v2'},indent=2))
