\
\
\
\
\
import os,json,time,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session12/action_embeddings')
SCRATCH=Path('cache')
TOK=SCRATCH/'seq_tokens';ACT=SCRATCH/'action_tokens';C=pl.col;torch.set_num_threads(3)
class ActEnc(nn.Module):
    def __init__(self):
        super().__init__();self.role=nn.Embedding(4,64);self.lrole=nn.Embedding(4,64);self.pos=nn.Embedding(24,64);self.inp=nn.Linear(24,64);layer=nn.TransformerEncoderLayer(64,4,dim_feedforward=128,dropout=.1,batch_first=True,norm_first=True);self.enc=nn.TransformerEncoder(layer,1);self.att=nn.Linear(64,1)
    def forward(self,x,ro,lr):
        pad=ro==3;h=self.inp(x)+self.role(ro)+self.lrole(lr)+self.pos(torch.arange(24))[None];allpad=pad.all(1);pad2=pad.clone();pad2[allpad,0]=False;h=self.enc(h,src_key_padding_mask=pad2);a=self.att(h).squeeze(-1).masked_fill(pad2,-1e4);out=(torch.softmax(a,1)[:,:,None]*h).sum(1);out[allpad]=0;return out
def encode(model,xa,ac,la,p1,p2):
    out=[]
    with torch.no_grad():
        for i in range(0,len(xa),128):
            x=torch.tensor(xa[i:i+128]);a=ac[i:i+128];l=la[i:i+128];v=[]
            for aa,bb in [(p1[i:i+128],p2[i:i+128]),(p2[i:i+128],p1[i:i+128])]:
                r=np.zeros_like(a,np.int64);r[a==aa[:,None]]=1;r[a==bb[:,None]]=2;r[a<0]=3;lr=np.ones_like(l,np.int64);lr[l<0]=0;lr[l==aa[:,None]]=2;lr[l==bb[:,None]]=3;v.append(model(x,torch.tensor(r),torch.tensor(lr)).numpy())
            out.append(np.column_stack([(v[0]+v[1])*.5,abs(v[0]-v[1])]))
    return np.concatenate(out)
def main():
    ROOT.mkdir(parents=True,exist_ok=True);cfg=json.load(open('artifacts/seq_v7/config.json'));assert cfg.get('aux_evidence',0)==0 and cfg['d_act']==64 and cfg['act_layers']==1;d=hand_data().select('pair_id','hand_id','table_id','time','fold');norm=np.load('artifacts/seq_v7/norm.npz');mu=norm['amu'];sd=norm['asd'];tables=sorted(p.stem for p in TOK.glob('*.npz'));sample=np.concatenate([np.asarray(np.load(ACT/f'{t}_XA.npy',mmap_mode='r')[::40]).reshape(-1,24).astype(np.float32) for t in tables[::8]]);sample=sample[abs(sample).sum(1)>0];np.testing.assert_allclose(sample.mean(0),mu,rtol=1e-5,atol=1e-6);np.testing.assert_allclose(sample.std(0)+1e-3,sd,rtol=1e-5,atol=1e-6);foldmap={t:f['fold'] for f in json.load(open('artifacts/folds.json')) for t in f['valid_tables']};assert all(foldmap[t]==f for t,f in d.select('table_id','fold').unique().iter_rows());torch.manual_seed(1212);random=ActEnc().eval();trained=[];hashes={}
    for f in range(4):
        path=Path(f'artifacts/seq_v7/seq_fold{f}.pt');state=torch.load(path,map_location='cpu',weights_only=True);m=ActEnc();m.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('act.')},strict=True);trained.append(m.eval());hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    parts=[[] for _ in range(4)];randomparts=[];start=time.time();done=0;swap_error=0.
    for (table,),q in d.group_by('table_id'):
        q=q.sort('pair_id','time','hand_id');z=np.load(TOK/f'{table}.npz');a=np.load(ACT/f'{table}.npz');off=np.r_[0,np.cumsum(z['n'])];rows=[];one=[];two=[]
        for (pid,),g in q.group_by('pair_id',maintain_order=True):
            kk=np.flatnonzero((z['pair_id']==pid)&(z['phase']=='development'));assert len(kk)==1;k=kk[0];lookup={h:int(j) for h,j in zip(z['hand_id'][off[k]:off[k+1]],a['hidx'][off[k]:off[k+1]])};rows.extend([lookup[h] for h in g['hand_id']]);one.extend([a['p1'][k]]*len(g));two.extend([a['p2'][k]]*len(g))
        raw=np.load(ACT/f'{table}_XA.npy',mmap_mode='r');xx=(np.asarray(raw[rows]).astype(np.float32)-mu)/sd;ac=a['ACT'][rows];la=a['LAG'][rows];one=np.array(one);two=np.array(two);key=q.select('pair_id','hand_id');r=encode(random,xx,ac,la,one,two);randomparts.append(key.with_columns(*[pl.Series(f'emb_{j}',r[:,j]) for j in range(128)]))
        for f,m in enumerate(trained):
            v=encode(m,xx,ac,la,one,two);parts[f].append(key.with_columns(*[pl.Series(f'emb_{j}',v[:,j]) for j in range(128)]))
            if done==0:swap_error=max(swap_error,float(np.max(abs(v[:8]-encode(m,xx[:8],ac[:8],la[:8],two[:8],one[:8])))))
        done+=1
        if done%40==0:print('action embeddings tables',done,'seconds',round(time.time()-start,1),flush=True)
    for f,parts_f in enumerate(parts):pl.concat(parts_f).write_parquet(ROOT/f'trained_outer{f}.parquet',compression='zstd')
    pl.concat(randomparts).write_parquet(ROOT/'random.parquet',compression='zstd');assert swap_error<1e-5;(ROOT/'audit.json').write_text(json.dumps({'rows_per_encoder':len(d),'dimensions':128,'normalization':'exact archived cache replay','held_out_pool_fold_mapping':'all evidence tables match sequence folds','feature_supervision':'pair risk only, no hand evidence loss','training_pool_features':'in-sample frozen pair encoder; validation pools excluded from that encoder','role_swap_max_error':swap_error,'checkpoint_sha256':hashes},indent=2));print('action embeddings complete',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
