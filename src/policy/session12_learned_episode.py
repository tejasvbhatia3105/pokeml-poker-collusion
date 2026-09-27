\
\
import os,json,time,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session11_conditional_family import Model,features,template
from session8_data import hand_data
from session8_count_conditioning import conditioned
ROOT=Path(os.environ.get('EPISODE_ROOT','artifacts/evidence_session12/learned_episode'));C=pl.col;torch.set_num_threads(1)
CONFIG={'steps':120,'lr':.03,'seeds':[1010,2020],'state_occupancy':.5,'initial_contrast':1.,'initial_tau':.08,'contrast_bound':3.,'penalty_contrast':.001,'penalty_bias':.01,'fit':'family-specific two-state emissions and time decay; frozen R30; average both seed likelihoods','control':'independent states with identical emission parameters'}
class Episode(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.bias=nn.Parameter(torch.zeros(3,2));self.contrast=nn.Parameter(torch.full((3,2),-np.log(2),dtype=torch.float64));self.scale=nn.Parameter(torch.full((3,),np.log(.08),dtype=torch.float64))
    def forward(self,prior,gaps,family,mask):
        bias=self.bias[family];amp=3*torch.sigmoid(self.contrast[family]);logits=prior[:,:,None,:]+bias[:,None,None,:]+amp[:,None,None,:]*torch.tensor([-1.,1.])[None,None,:,None];p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:,:,:1]),logits],3),3);fallback=torch.zeros_like(p);fallback[:,:,:,0]=1;p=torch.where(mask[:,:,None,None],p,fallback)
        rho=torch.exp(-gaps/self.scale[family,None].exp().clamp(.001,2)) if self.kind=='episode' else torch.zeros_like(gaps);T=(1-rho[:,:,None,None])*.5+rho[:,:,None,None]*torch.eye(2)[None,None];return p,T

def loss(p,T,templates,valid,den,minimum):
    B,n=p.shape[:2];z=torch.full((B,6,2),.5,dtype=p.dtype);ll=torch.zeros((B,6),dtype=p.dtype);count=torch.zeros((B,6,2),dtype=p.dtype);count[:,0]=.5
    for i in range(n):
        z=torch.einsum('bks,bsr->bkr',z,T[:,i]);terms=torch.stack([torch.ones_like(p[:,i,:,0]),p[:,i,:,1],p[:,i,:,2],p[:,i,:,0],p[:,i,:,0]+p[:,i,:,2]],1);v=terms.gather(1,templates[:,:,i,None].expand(-1,-1,2));z=z*v;scale=z.sum(2).clamp_min(1e-100);ll=ll+scale.log();z=z/scale[:,:,None]
        count=torch.einsum('bks,bsr->bkr',count,T[:,i]);event=p[:,i,:,1]+p[:,i,:,2];count=torch.cat([count[:,:1]*p[:,i,None,:,0],count[:,1:5]*p[:,i,None,:,0]+count[:,:4]*event[:,None],count[:,5:]+count[:,4:5]*event[:,None]],1)
    observed=torch.logsumexp(ll.masked_fill(~valid,-1e6),1);normal=(count.sum(2)*(torch.arange(6)[None]>=minimum[:,None])).sum(1).clamp_min(1e-100).log();return (-observed+normal)/den

def inclusion(p,T,minimum):
    \
    p=np.asarray(p,dtype=np.float64);T=np.asarray(T,dtype=np.float64);n=len(p);a=p[:,:,1];b=p[:,:,2];s=a+b
    def forward(c):
        z=np.zeros((n,6,2));z[0,0]=.5
        for i in range(n-1):
            after=z[i]*(1-c[i]);after[1:5]+=z[i,:4]*c[i];after[5]+=z[i,4]*c[i]+z[i,5]*c[i];z[i+1]=after@T[i+1]
        return z
    def backward(c):
        z=np.zeros((n,6,2));z[-1,0]=1
        for i in range(n-2,-1,-1):
            v=z[i+1]*(1-c[i+1]);v[1:5]+=z[i+1,:4]*c[i+1];v[5]+=z[i+1,4]*c[i+1]+z[i+1,5]*c[i+1];z[i]=v@T[i+1].T
        return z
    fp=forward(a);fa=forward(s);bp=backward(a);ba=backward(s);end=fa[-1]*(1-s[-1]);end[1:5]+=fa[-1,:4]*s[-1];end[5]+=fa[-1,4]*s[-1]+fa[-1,5]*s[-1];mass=end[minimum:].sum();out=[]
    for i in range(n):
        raw=(fp[i,:5]*a[i]).sum()+sum((fa[i,k]*b[i]*bp[i,:5-k].sum(0)).sum() for k in range(5));short=sum((fa[i,k]*s[i]*ba[i,:minimum-1-k].sum(0)).sum() for k in range(minimum-1));out.append((raw-short)/mass)
    out=np.array(out);assert out.min()>-1e-6 and out.max()<1+1e-6 and out.sum()<=5+1e-6;return np.clip(out,0,1)

def check():
    rng=np.random.default_rng(121212);n=6;p=rng.dirichlet([3,1,1],size=(n,2));T=np.tile(np.array([[.8,.2],[.2,.8]]),(n,1,1));T[0]=np.eye(2);truth={};mass={k:0. for k in [3,5]};inc={k:np.zeros(n) for k in mass}
    for states in itertools.product(range(2),repeat=n):
        ps=.5*np.prod([T[i,states[i-1],states[i]] for i in range(1,n)])
        for cats in itertools.product(range(3),repeat=n):
            pr=ps*np.prod(p[np.arange(n),states,cats]);e=tuple(([i for i,c in enumerate(cats) if c==1]+[i for i,c in enumerate(cats) if c==2])[:5]);truth[e]=truth.get(e,0)+pr
            for k in mass:
                if len(e)>=k:mass[k]+=pr;inc[k][list(e)]+=pr
    err=0.;likerr=0.
    for k in mass:
        err=max(err,float(abs(inclusion(p,T,k)-inc[k]/mass[k]).max()))
        for e,pr in truth.items():
            if len(e)<k:continue
            z=template(n,np.array(e));tt=np.zeros((1,6,n),np.int64);vv=np.zeros((1,6),bool);tt[0,:len(z)]=z;vv[0,:len(z)]=True;l=loss(torch.tensor(p)[None],torch.tensor(T)[None],torch.tensor(tt),torch.tensor(vv),torch.tensor([len(e)]),torch.tensor([k]));likerr=max(likerr,abs(np.exp(-l.item()*len(e))-pr/mass[k]))
    independent=np.tile(p.mean(1)[:,None,:],(1,2,1));inderr=max(float(abs(inclusion(independent,T,k)-conditioned(independent[:,0,1:],k)).max()) for k in mass);assert max(err,likerr,inderr)<1e-11
    return {'hidden_and_category_paths':6**n,'inclusion_error':err,'likelihood_error':likerr,'independent_limit_error':inderr}

def main():
    from session12_hmm_loss import fast_loss,check as gradient_check
    fit_loss=fast_loss if os.environ.get('EPISODE_FAST')=='1' else loss
    ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));checks=check()
    if os.environ.get('EPISODE_FAST')=='1':checks['forward_backward_gradients']=gradient_check()
    (ROOT/'mathematical_checks.json').write_text(json.dumps(checks,indent=2));print(checks,flush=True);d=hand_data();parts=[];audit=[];start=time.time()
    for f in range(4):
        q=d.join(pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');bags=[]
        for (pid,),g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');x,p=features(g);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];bags.append({'pid':pid,'g':g,'x':x,'p':p,'template':template(len(g),e),'den':len(e),'fold':g['fold'][0],'family':['directed_transfer','soft_play','coordinated_isolation'].index(g['behavior_family'][0])})
        bags.sort(key=lambda b:b['pid']);N=len(bags);n=max(len(b['g']) for b in bags);X=np.zeros((N,n,35),np.float32);P=np.zeros((N,n,2),np.float32);M=np.zeros((N,n),bool);G=np.zeros((N,n));TT=np.zeros((N,6,n),np.int64);V=np.zeros((N,6),bool);D=np.array([b['den'] for b in bags]);F=np.array([b['family'] for b in bags]);fv=np.array([b['fold'] for b in bags])
        for i,b in enumerate(bags):
            k=len(b['g']);X[i,:k]=b['x'];P[i,:k]=b['p'];M[i,:k]=True;G[i,1:k]=np.diff(b['g']['relative_time']);t=b['template'];TT[i,:len(t),:k]=t;V[i,:len(t)]=True
        tr=np.flatnonzero((fv!=f)&V.any(1));va=np.flatnonzero(fv==f);priors=[]
        for seed in CONFIG['seeds']:
            state=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);base=Model(35,'independent');base.load_state_dict(state['state_dict']);base.eval()
            with torch.no_grad():prior=torch.tensor(P)+base(torch.tensor(np.clip((X-state['mu'])/state['sd'],-6,6)),torch.tensor(M));priors.append(prior.double())
        K=np.array([state['minimums'][b['g']['behavior_family'][0]] for b in bags]);idx=np.tile(tr,2);PP=torch.cat([p[tr] for p in priors]);GG=torch.tensor(G[idx]);FF=torch.tensor(F[idx]);MM=torch.tensor(M[idx]);templates=torch.tensor(TT[idx]);valid=torch.tensor(V[idx]);den=torch.tensor(D[idx]);minimum=torch.tensor(K[idx]);outputs={}
        for kind in ['independent','episode']:
            model=Episode(kind).double();opt=torch.optim.Adam(model.parameters(),lr=CONFIG['lr']);trace=[]
            for step in range(CONFIG['steps']):
                p,t=model(PP,GG,FF,MM);ll=fit_loss(p,t,templates,valid,den,minimum).mean();reg=CONFIG['penalty_contrast']*(3*torch.sigmoid(model.contrast)).square().mean()+CONFIG['penalty_bias']*model.bias.square().mean();value=ll+reg;assert torch.isfinite(value);opt.zero_grad();value.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();trace.append(float(value.detach()))
                if step%30==0:print('episode fit',f,kind,step,round(trace[-1],6),'seconds',round(time.time()-start,1),flush=True)
            torch.save({'state_dict':model.state_dict(),'fold':f,'kind':kind,'minimums':state['minimums'],'config':CONFIG},ROOT/f'{kind}_fold{f}.pt');pred=[]
            with torch.no_grad():
                for prior in priors:
                    p,t=model(prior[va],torch.tensor(G[va]),torch.tensor(F[va]),torch.tensor(M[va]));pred.append((p.numpy(),t.numpy()))
            outputs[kind]=pred;audit.append({'fold':f,'kind':kind,'train_pairs':len(tr),'trace':trace,'parameters':{k:v.detach().numpy().tolist() for k,v in model.state_dict().items()}})
        for j,i in enumerate(va):
            g=bags[i]['g'];k=len(g);z=g.select('pair_id','hand_id')
            for kind,pred in outputs.items():
                inc=np.mean([inclusion(p[j,:k],t[j,:k],K[i]) for p,t in pred],0);z=z.with_columns(pl.Series(kind,.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc))
            parts.append(z)
        (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));pl.concat(parts).write_parquet(ROOT/'partial_oof.parquet')
    pl.concat(parts).write_parquet(ROOT/'episode_oof.parquet')
if __name__=='__main__':main()
