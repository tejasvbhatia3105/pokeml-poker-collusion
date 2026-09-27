\
\
\
\
\
\
\
\
import os,json,time,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session8_data import hand_data
from session9_selector import PHYSICAL,FAMILIES,logit
from session6_priority import inclusion
ROOT=Path(os.environ.get('LIST_OUTPUT_ROOT','artifacts/evidence_session12/frozen_'+os.environ.get('EMBED_KIND','trained')));C=pl.col;torch.set_num_threads(3)
CONFIG={'seeds':[1010,2020],'epochs':80,'batch_size':16,'learning_rate':.0007,'weight_decay':.01,'residual_l2':.02,'residual_bound':1.5,'features':'35 compact R29 score, physical, chronological and family inputs','loss':'exact ordered two-tier capped-list likelihood, marginalized over valid splits; divide by truth count','selection':'Fixed schedule; independent and contextual correction; no held-out checkpoint choice','prediction':'.25 R27 base + .25 R28 Cat inclusion + .5 corrected joint Cat/HGB inclusion'}
def template(n,e,cap=5):
    out=[]
    for k in range(len(e)+1):
        if np.any(np.diff(e[:k])<=0) or np.any(np.diff(e[k:])<=0):continue
        if len(e)<cap:z=np.full(n,3)
        elif k==cap:z=np.where(np.arange(n)<=e[k-1],4,0)
        else:z=np.where(np.arange(n)<=e[-1],3,4)
        z[e[:k]]=1;z[e[k:]]=2;out.append(z)
    return np.array(out,dtype='int64').reshape(-1,n)
def list_nll(logits,templates,valid,den):
    lp=torch.log_softmax(torch.cat([torch.zeros_like(logits[:,:,:1]),logits],2),2)
    terms=torch.stack([torch.zeros_like(lp[:,:,0]),lp[:,:,1],lp[:,:,2],lp[:,:,0],torch.logsumexp(lp[:,:,[0,2]],2)],2)
    selected=terms[:,None].expand(-1,templates.shape[1],-1,-1).gather(3,templates[:,:,:,None]).squeeze(-1).sum(2)
    selected=selected.masked_fill(~valid,-1e6)
    return -torch.logsumexp(selected,1)/den
def features(g):
    n=len(g);s=g.select('base','cat_primary','cat_secondary','hist_primary','hist_secondary','cat_inclusion','joint_inclusion','r29').to_numpy();cp=s[:,1:3]/np.maximum(1,s[:,1:3].sum(1))[:,None];hp=s[:,3:5]/np.maximum(1,s[:,3:5].sum(1))[:,None];jp=.5*(cp+hp);rank=np.lexsort((g['hand_id'].to_numpy(),-s[:,7]));rr=np.empty(n);rr[rank]=np.arange(n)/max(1,n-1)
    a=cp[:,0];b=jp[:,0];extra=np.column_stack([g['relative_time'],np.arange(n)/max(1,n-1),np.full(n,np.log1p(n)),rr,(np.cumsum(a)-a)/5,(a.sum()-np.cumsum(a))/5,(np.cumsum(b)-b)/5,(b.sum()-np.cumsum(b))/5,np.full(n,a.sum()/5),np.full(n,b.sum()/5)]);physical=g.select(PHYSICAL).to_numpy();physical=np.sign(physical)*np.log1p(abs(physical));family=np.tile(np.eye(3)[FAMILIES.index(g['behavior_family'][0])],(n,1));X=np.column_stack([logit(s),extra,physical,family]).astype('float32');prior=np.log(np.maximum(jp,1e-6))-np.log(np.maximum(1-jp.sum(1),1e-6))[:,None]
    return X,prior.astype('float32')
base_features=features
def features(g):
    x,p=base_features(g)
    return np.column_stack([x,g.select([f'emb_{i}' for i in range(128)]).to_numpy()]).astype('float32'),p
CONFIG['features']='35 R30 compact inputs plus 128 frozen action encoder role-invariant coordinates'
CONFIG['embedding_kind']=os.environ.get('EMBED_KIND','trained')
assert CONFIG['embedding_kind'] in ['trained','random']
class Model(nn.Module):
    def __init__(self,features,kind):
        super().__init__();self.kind=kind;self.local=nn.Sequential(nn.Linear(features,48),nn.GELU(),nn.Dropout(.1),nn.Linear(48,32),nn.GELU())
        if kind=='contextual':self.attention=nn.MultiheadAttention(32,4,dropout=.1,batch_first=True);self.norm=nn.LayerNorm(32)
        self.head=nn.Linear(32,2);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x,mask):
        h=self.local(x)
        if self.kind=='contextual':v,_=self.attention(h,h,h,key_padding_mask=~mask,need_weights=False);h=self.norm(h+v)
        return CONFIG['residual_bound']*torch.tanh(self.head(h))*mask[:,:,None]
def check():
    rng=np.random.default_rng(1001);p=rng.dirichlet([3,1,1],size=6);found={}
    for categories in itertools.product(range(3),repeat=6):
        chosen=tuple(([i for i,k in enumerate(categories) if k==1]+[i for i,k in enumerate(categories) if k==2])[:5]);found[chosen]=found.get(chosen,0)+np.prod(p[np.arange(6),categories])
    error=0.
    for e,expected in found.items():
        if len(e)<1:continue
        z=template(6,np.array(e));logits=torch.tensor(np.log(p[:,1:]/p[:,:1]))[None];nll=list_nll(logits,torch.tensor(z)[None],torch.ones((1,len(z)),dtype=torch.bool),torch.tensor([len(e)]));actual=np.exp(-nll.item()*len(e));error=max(error,abs(actual-expected))
    assert error<1e-12
    (ROOT/'list_likelihood_checks.json').write_text(json.dumps({'enumerated_category_paths':3**6,'observed_lists_checked':len(found)-1,'maximum_probability_error':error},indent=2))
def log_count_at_least(logits,mask,minimum):
    lp=torch.log_softmax(torch.cat([torch.zeros_like(logits[:,:,:1]),logits],2),2);p=(1-lp[:,:,0].exp())*mask
    z=torch.zeros((len(p),6),dtype=p.dtype,device=p.device);z[:,0]=1
    for i in range(p.shape[1]):
        v=p[:,i:i+1];z=torch.cat([z[:,:1]*(1-v),z[:,1:5]*(1-v)+z[:,:4]*v,z[:,5:6]+z[:,4:5]*v],1)
    return torch.log((z*(torch.arange(6)[None]>=minimum[:,None])).sum(1).clamp_min(1e-15))

def conditional_nll(logits,templates,valid,den,mask,minimum):
    return list_nll(logits,templates,valid,den)+log_count_at_least(logits,mask,minimum)/den

CONFIG['conditioning']='Ascertainment hypothesis: family minimum listed count estimated on compatible outer-training pairs only; conditional training and selection'
CONFIG['loss']='exact ordered capped-list likelihood conditional on outer-training family minimum, divided by truth count'
from session8_count_conditioning import conditioned

def main():
    ROOT.mkdir(exist_ok=True);path=ROOT/'list_learning_config.json'
    if path.exists():assert json.loads(path.read_text())==CONFIG
    else:path.write_text(json.dumps(CONFIG,indent=2))
    input_root=Path(os.environ.get('LIST_INPUT_ROOT','artifacts/evidence_session10/nested6'));kinds=os.environ.get('LIST_KINDS','independent').split(',');assert set(kinds)<=set(['independent','contextual']);(ROOT/'list_run.json').write_text(json.dumps({'input_root':str(input_root),'kinds':kinds},indent=2));check();d=hand_data();allparts=[];audit=[];start=time.time()
    for f in range(4):
        q=d.join(pl.read_parquet(input_root/f'nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');embedding_file=Path('artifacts/evidence_session12/action_embeddings')/(f'trained_outer{f}.parquet' if CONFIG['embedding_kind']=='trained' else 'random.parquet');q=q.join(pl.read_parquet(embedding_file),on=['pair_id','hand_id'],validate='1:1');bags=[]
        for (pid,),g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');x,prior=features(g);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];bags.append({'g':g,'pid':pid,'fold':g['fold'][0],'X':x,'prior':prior,'template':template(len(g),e),'den':len(e)})
        bags.sort(key=lambda b:b['pid']);n=max(len(b['g']) for b in bags);N=len(bags);nf=bags[0]['X'].shape[1];X=np.zeros((N,n,nf),dtype='float32');P=np.zeros((N,n,2),dtype='float32');M=np.zeros((N,n),bool);T=np.zeros((N,6,n),dtype='int64');V=np.zeros((N,6),bool);D=np.array([b['den'] for b in bags],dtype='float32');fv=np.array([b['fold'] for b in bags])
        for i,b in enumerate(bags):
            k=len(b['g']);X[i,:k]=b['X'];P[i,:k]=b['prior'];M[i,:k]=True;t=b['template'];T[i,:len(t),:k]=t;V[i,:len(t)]=True
        tr=(fv!=f)&V.any(1);va=fv==f;mu=X[tr][M[tr]].mean(0);sd=np.maximum(.05,X[tr][M[tr]].std(0));XX=torch.tensor(np.clip((X-mu)/sd,-6,6));PP=torch.tensor(P);MM=torch.tensor(M);TT=torch.tensor(T);VV=torch.tensor(V);DD=torch.tensor(D);train=np.flatnonzero(tr);valid=np.flatnonzero(va);pred={};families=np.array([b['g']['behavior_family'][0] for b in bags]);minimums={fam:int(D[tr&(families==fam)].min()) for fam in set(families)};KK=torch.tensor([minimums[fam] for fam in families])
        assert np.isfinite(X).all() and not np.any(tr&va)
        for kind in kinds:
            seedpred=[];losses=[]
            for seed in CONFIG['seeds']:
                torch.manual_seed(seed+f);rng=np.random.default_rng(seed+f);m=Model(nf,kind);opt=torch.optim.AdamW(m.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay']);m.eval()
                with torch.no_grad():assert torch.count_nonzero(m(XX[valid[:2]],MM[valid[:2]])).item()==0
                for epoch in range(CONFIG['epochs']):
                    m.train();order=rng.permutation(train)
                    for j in range(0,len(order),CONFIG['batch_size']):
                        ix=order[j:j+CONFIG['batch_size']];delta=m(XX[ix],MM[ix]);loss=conditional_nll(PP[ix]+delta,TT[ix],VV[ix],DD[ix],MM[ix],KK[ix]).mean()+CONFIG['residual_l2']*delta.square().sum()/MM[ix].sum()/2;assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                m.eval()
                with torch.no_grad():
                    logits=PP[valid]+m(XX[valid],MM[valid]);prob=torch.softmax(torch.cat([torch.zeros_like(logits[:,:,:1]),logits],2),2).numpy();seedpred.append(prob);losses.append(float(loss))
                torch.save({'state_dict':m.state_dict(),'mu':mu,'sd':sd,'feature_count':nf,'kind':kind,'fold':f,'seed':seed,'config':CONFIG,'minimums':minimums},ROOT/f'list_{kind}_fold{f}_seed{seed}.pt')
            pred[kind]=seedpred;audit.append({'fold':f,'kind':kind,'train_pairs':len(train),'excluded_incompatible_train_lists':int(((fv!=f)&~V.any(1)).sum()),'valid_pairs':len(valid),'final_batch_losses':losses});print('list learning',f,kind,round(time.time()-start,1),flush=True)
        for k,i in enumerate(valid):
            b=bags[i];g=b['g'];z=g.select('pair_id','hand_id','fold','evidence','r29');count=len(g)
            for kind in pred:
                inc=np.mean([conditioned(p[k,:count,1:3],minimums[g['behavior_family'][0]]) for p in pred[kind]],axis=0);new=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;z=z.with_columns(pl.Series(kind,new),pl.Series(kind+'_inclusion',inc))
            allparts.append(z)
    pl.concat(allparts).write_parquet(ROOT/'list_learning_oof.parquet');(ROOT/'list_learning_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
