\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,time,hashlib,itertools,sys
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
from sklearn.tree import DecisionTreeRegressor
import session11_list_boost as base
from session62_grounded_list_boost import grounded
from session117_fast_count import log_count_at_least,provenance
from session8_count_conditioning import conditioned
from session10_list_learning import list_nll,template
from session8_data import hand_data

ROOT=Path('artifacts/evidence_session130_hierarchical_list')
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
CONFIG=dict(base.CONFIG, input_root='artifacts/evidence_session55_current_nested',
            kinds=['point','random_rate'],quadrature=9,rate_bound=2.,rate_initial=.5,
            rate_fit_every=10,rate_maxiter=30,rate_parameterization='variance with bounded three-point finite-difference derivatives',
            hypothesis='One Gaussian log-odds intercept shared by all hands and both event tiers in a pair.',
            conditioning='Integrate list likelihood and count-tail likelihood separately, then form their ratio.',
            regularization='Original squared correction penalty plus its expected 2*sigma^2 rate contribution.',
            selection='Fixed full-feature schedule; no heldout stopping, weight search or family splice.')
torch.set_num_threads(3)
QX,QW=np.polynomial.hermite.hermgauss(CONFIG['quadrature']);QX*=np.sqrt(2);QW/=np.sqrt(np.pi)

def pack(d,f):
    base.CONFIG=dict(base.CONFIG,input_root=CONFIG['input_root'])
    cols=[c for c in d.columns if c.startswith('grounded_')]
    values=base.pack(d,f,cols)
    groups,x,big,P,M,T,V,D,fv,bid,pos=values
    fam=np.array([FAMILIES.index(g['behavior_family'][0]) for g in groups])
    tr=np.flatnonzero((fv!=f)&V.any(1).numpy())
    minimums={j:int(D.numpy()[tr[fam[tr]==j]].min()) for j in range(3)}
    K=torch.tensor([minimums[j] for j in fam]);return values,fam,K,tr

def mixture_nll(z,M,T,V,D,K,sigma):
                                                                      
    q=torch.tensor(QX,dtype=z.dtype);lw=torch.tensor(np.log(QW),dtype=z.dtype)
    logits=(z[None]+q[:,None,None,None]*sigma[None,:,None,None]).flatten(0,1)
    B=len(z);Q=len(q)
    ll=-list_nll(logits,T.repeat(Q,1,1),V.repeat(Q,1),D.repeat(Q)).reshape(Q,B)*D[None]
    tail=log_count_at_least(logits,M.repeat(Q,1),K.repeat(Q)).reshape(Q,B)
    return (-torch.logsumexp(ll+lw[:,None],0)+torch.logsumexp(tail+lw[:,None],0))/D

def objective(P,A,M,T,V,D,K,fam,train,sigma,kind):
    z=P[train]+A[train]
    if kind=='point':nll=list_nll(z,T[train],V[train],D[train])+log_count_at_least(z,M[train],K[train])/D[train]
    else:nll=mixture_nll(z,M[train],T[train],V[train],D[train],K[train],sigma[fam[train]])
    reg=(A[train].square().sum(2)*M[train]).sum(1)/M[train].sum(1)
    return nll.sum()+CONFIG['ridge']*(reg+2*sigma[fam[train]].square()).sum()

def rate_opt(P,A,M,T,V,D,K,fam,tr,sigma):
    def f(v):
        with torch.no_grad():
            s=torch.tensor(np.sqrt(v),dtype=P.dtype)
            return float(objective(P,A,M,T,V,D,K,fam,tr,s,'random_rate'))
                                                                           
                                                                            
                                                                        
    result=minimize(f,sigma.numpy().astype(float)**2,jac='3-point',method='L-BFGS-B',bounds=[(0,CONFIG['rate_bound']**2)]*3,
                    options=dict(maxiter=CONFIG['rate_maxiter'],ftol=1e-9,gtol=1e-5,maxls=20,finite_diff_rel_step=1e-4))
    return torch.tensor(np.sqrt(result.x),dtype=P.dtype),dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))

def inclusion_mixture(z,k,sigma):
    pp=softmax(np.concatenate([np.zeros((len(QX),len(z),1)),z[None]+QX[:,None,None]*sigma],2),axis=2)
    zz=torch.tensor(np.log(pp[:,:,1:]/pp[:,:,:1]),dtype=torch.float64)
    tail=log_count_at_least(zz,torch.ones(zz.shape[:2],dtype=torch.bool),torch.full((len(QX),),k)).numpy()
    weights=softmax(np.log(QW)+tail)
    inc=np.stack([conditioned(p[:,1:],k) for p in pp])
    return weights@inc

def check():
    rng=np.random.default_rng(130);n=6;p=rng.dirichlet([4,1,1],n);z=np.log(p[:,1:]/p[:,:1]);sigma=.65;k=3
    enumeration={};den=0.;num=np.zeros(n)
    for node,w in zip(QX,QW):
        pp=softmax(np.column_stack([np.zeros(n),z+node*sigma]),1)
        for cats in itertools.product(range(3),repeat=n):
            prob=w*np.prod(pp[np.arange(n),cats]);ev=[i for i,c in enumerate(cats) if c==1]+[i for i,c in enumerate(cats) if c==2]
            if len(ev)>=k:
                den+=prob;num[ev[:5]]+=prob;key=tuple(ev[:5]);enumeration[key]=enumeration.get(key,0)+prob
    pred=inclusion_mixture(z,k,sigma);incerr=float(abs(pred-num/den).max());assert incerr<1e-10
    errors=[]
    for e,prob in list(enumeration.items())[::7]:
        t=template(n,np.array(e));a=mixture_nll(torch.tensor(z)[None],torch.ones((1,n),dtype=torch.bool),torch.tensor(t)[None],torch.ones((1,len(t)),dtype=torch.bool),torch.tensor([len(e)]),torch.tensor([k]),torch.tensor([sigma],dtype=torch.float64))
        errors.append(abs(np.exp(-float(a)*len(e))-prob/den))
    assert max(errors)<1e-10
    e=list(enumeration)[len(enumeration)//2];t=torch.tensor(template(n,np.array(e)))[None];v=torch.ones(t.shape[:2],dtype=torch.bool)
    zz=torch.tensor(z)[None].requires_grad_(True);s=torch.tensor([sigma],dtype=torch.float64,requires_grad=True)
    fn=lambda a,b:mixture_nll(a,torch.ones((1,n),dtype=torch.bool),t,v,torch.tensor([len(e)]),torch.tensor([k]),b)
    assert torch.autograd.gradcheck(fn,(zz,s),eps=1e-6,atol=1e-5)
    out=dict(quadrature_paths=len(QX)*3**n,conditional_lists_checked=len(errors),inclusion_max_error=incerr,list_probability_max_error=max(errors),logit_and_sigma_gradcheck=True)
    ROOT.mkdir(exist_ok=True);(ROOT/'math_checks.json').write_text(json.dumps(out,indent=2));print(out,flush=True)

def data():
    d=hand_data();x,c=grounded(d)
    old=np.load('artifacts/evidence_session62_grounded_list_boost/grounded_features.npz')['x']
    assert np.array_equal(x,old), 'Rebuilt grounded inputs differ from archived strong control'
    return d.with_columns(*[pl.Series(n,x[:,i]) for i,n in enumerate(c)])

def train():
    ROOT.mkdir(exist_ok=True);assert not list(ROOT.glob('model_*.joblib')),'Do not overwrite completed models'
    (ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));(ROOT/'source.json').write_text(json.dumps({'sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'count_provenance':provenance()},indent=2))
    check();d=data();t0=time.time();audit=[];parts=[]
    for f in range(4):
        (groups,x,big,P,M,T,V,D,fv,bid,pos),fam,K,tr=pack(d,f)
        P=P.double();D=D.double();X=np.nan_to_num(np.column_stack([x,big]),nan=0,posinf=1e6,neginf=-1e6)
        rt=np.isin(bid,tr);va=np.flatnonzero(fv==f);rowweight=1/D.numpy()[bid]
        for kind in CONFIG['kinds']:
            A=torch.zeros_like(P);sigma=torch.full((3,),0. if kind=='point' else CONFIG['rate_initial'],dtype=P.dtype)
            steps=[];history=[];rates=[]
            def loss(a,s=sigma):return objective(P,a,M,T,V,D,K,fam,tr,s,kind)
            initial=float(loss(A))
            for it in range(CONFIG['iterations']):
                if kind=='random_rate' and it%CONFIG['rate_fit_every']==0:
                    sigma,info=rate_opt(P,A,M,T,V,D,K,fam,tr,sigma);rates.append(dict(iteration=it,sigma=sigma.tolist(),optimizer=info))
                A.requires_grad_(True);value=objective(P,A,M,T,V,D,K,fam,tr,sigma,kind);value.backward()
                grad=A.grad.numpy()[bid,pos];a=A.detach().numpy()[bid,pos]
                pr=torch.softmax(torch.cat([torch.zeros_like(P[:,:,:1]),P+A.detach()],2),2).numpy()[bid,pos,1:]
                h=np.maximum(pr*(1-pr),.02)*rowweight[:,None]+2*CONFIG['ridge']/M.sum(1).numpy()[bid,None]
                target=np.clip(-grad/h,-5,5)
                tree=DecisionTreeRegressor(max_leaf_nodes=CONFIG['max_leaf_nodes'],min_samples_leaf=CONFIG['min_samples_leaf'],max_features=CONFIG['max_features'],random_state=CONFIG['seed']+f*1000+it)
                tree.fit(X[rt],target[rt],sample_weight=h[rt].mean(1));update=tree.predict(X);old=float(value.detach());A=A.detach();step=CONFIG['learning_rate']
                accepted=False
                for _ in range(7):
                    candidate=torch.zeros_like(P);candidate[bid,pos]=torch.tensor(np.clip(a+step*update,-CONFIG['bound'],CONFIG['bound']))
                    new=float(objective(P,candidate,M,T,V,D,K,fam,tr,sigma,kind))
                    if new<=old+1e-6:accepted=True;break
                    step*=.5
                if not accepted:step=0.;candidate=A;new=old
                A=candidate;steps.append((tree,step));history.append(new)
                if it%40==0:print(f,kind,it,'loss',round(new,3),'sigma',sigma.tolist(),'seconds',round(time.time()-t0),flush=True)
            model=dict(steps=steps,sigma=sigma.numpy(),kind=kind,fold=f,config=CONFIG,train_pairs=[groups[i]['pair_id'][0] for i in tr],columns=[c for c in d.columns if c.startswith('grounded_')])
            path=ROOT/f'model_{kind}_fold{f}.joblib';joblib.dump(model,path,compress=3)
            check_model=joblib.load(path);delta=infer(check_model,X);assert np.max(abs(delta-A.numpy()[bid,pos]))<1e-10
            for i in va:
                g=groups[i];z=P[i,:len(g)].numpy()+A[i,:len(g)].numpy()
                inc=inclusion_mixture(z,int(K[i]),float(sigma[fam[i]]))
                score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
                parts.append(g.select('pair_id','hand_id','fold','evidence').with_columns(pl.lit(kind).alias('kind'),pl.Series('score',score),pl.Series('inclusion',inc)))
            rec=dict(fold=f,kind=kind,train_pairs=len(tr),validation_pairs=len(va),initial_loss=initial,losses=history,rate_updates=rates,sigma=sigma.tolist(),feature_sha=hashlib.sha256(X.tobytes()).hexdigest())
            audit.append(rec);(ROOT/'training.json').write_text(json.dumps(audit,indent=2));pl.concat(parts).write_parquet(ROOT/'oof.parquet');print('completed',f,kind,'seconds',round(time.time()-t0),flush=True)

def infer(model,X):
                                                                           
    a=np.zeros((len(X),2))
    for tree,step in model['steps']:a=np.clip(a+step*tree.predict(X),-CONFIG['bound'],CONFIG['bound'])
    return a

def verify():
    d=data();saved=pl.read_parquet(ROOT/'oof.parquet');audit=[]
    for f in range(4):
        (groups,x,big,P,M,T,V,D,fv,bid,pos),fam,K,tr=pack(d,f);X=np.nan_to_num(np.column_stack([x,big]),nan=0,posinf=1e6,neginf=-1e6)
        for kind in CONFIG['kinds']:
            model=joblib.load(ROOT/f'model_{kind}_fold{f}.joblib');assert model['train_pairs']==[groups[i]['pair_id'][0] for i in tr]
            assert not {groups[i]['table_id'][0] for i in tr}&{g['table_id'][0] for g in groups if g['fold'][0]==f}
            a=infer(model,X);reverse=infer(model,X[::-1])[::-1];assert np.array_equal(a,reverse)
            err=0.
            for i in np.flatnonzero(fv==f):
                g=groups[i];ix=np.flatnonzero(bid==i);z=P[i,:len(g)].double().numpy()+a[ix]
                inc=inclusion_mixture(z,int(K[i]),float(model['sigma'][fam[i]]));score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
                q=g.select('pair_id','hand_id').join(saved.filter(pl.col('kind')==kind),on=['pair_id','hand_id'],maintain_order='left',validate='1:1');err=max(err,float(abs(score-q['score'].to_numpy()).max()))
            assert err<1e-10
            audit.append(dict(fold=f,kind=kind,prediction_replay_max_error=err,query_reversal=True,training_validation_pools_disjoint=True))
            print('verified',f,kind,err,flush=True)
    (ROOT/'verification.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':{'train':train,'verify':verify,'check':check}[sys.argv[1]]()
