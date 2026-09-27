\
\
\
\
\
import os,json,time,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from session10_list_learning import template,ROOT
from session8_data import hand_data
from session6_priority import inclusion
C=pl.col
CONFIG={'initial':[.35,.5,150.],'bounds':[[.1,.9],[0,.98],[10,1500]],'maxiter':140,'loss':'nested training ordered-list NLL divided by truth count','regularization':'.01 contrast squared','selection':'One fixed bounded Nelder-Mead fit per family/fold, with independent null retained if training objective is lower','prior':'Per-pair mean unary event probability; empirical Bayes, not known generator prior'}
def parameters(v):return float(v[0]),float(v[1]),float(v[2])
def weights(p,rho,contrast):
    theta=np.clip(p[:,1:].sum(1).mean(),1e-5,1-1e-5);distance=contrast*min(theta/rho,(1-theta)/(1-rho));rates=np.array([theta-rho*distance,theta+(1-rho)*distance]);ratio=np.column_stack([(1-rates)/(1-theta),rates/theta,rates/theta]);return p[:,None,:]*ratio[None,:,:]
def transitions(t,rho,tau):
    r=np.exp(-np.diff(t)/tau);station=np.array([1-rho,rho]);return r[:,None,None]*np.eye(2)+(1-r[:,None,None])*station[None,None,:]
def observed_loss(bags,v):
    rho,contrast,tau=parameters(v);B=len(bags);N=max(len(b['p']) for b in bags);W=np.ones((B,N,2,5));T=np.broadcast_to(np.eye(2),(B,N,2,2)).copy();codes=np.zeros((B,7,N),dtype=int);valid=np.zeros((B,6),bool);den=np.array([b['den'] for b in bags]);station=np.array([1-rho,rho])
    for i,b in enumerate(bags):
        n=len(b['p']);w=weights(b['p'],rho,contrast);total=w.sum(2);W[i,:n]=np.stack([total,w[:,:,1],w[:,:,2],w[:,:,0],w[:,:,[0,2]].sum(2)],2);T[i,:n-1]=transitions(b['t'],rho,tau);z=b['template'];codes[i,1:1+len(z),:n]=z;valid[i,:len(z)]=True
    state=np.broadcast_to(station,(B,7,2)).copy();ll=np.zeros((B,7));bi=np.arange(B)[:,None];si=np.arange(2)[None,None,:]
    for j in range(N):
        emission=W[:,j][bi[:,:,None],si,codes[:,:,j,None]];state*=emission;scale=np.maximum(state.sum(2),1e-250);ll+=np.log(scale);state/=scale[:,:,None];state=np.einsum('bks,bst->bkt',state,T[:,j])
    val=logsumexp(np.where(valid,ll[:,1:],-1e250),1)-ll[:,0];return float(np.mean(-val/den)+.01*contrast**2)
def posterior(p,t,v):
    rho,contrast,tau=parameters(v);w=weights(p,rho,contrast);em=w.sum(2);q=w/em[:,:,None];T=transitions(t,rho,tau);n=len(p);back=np.ones((n,2))
    for i in range(n-2,-1,-1):
        back[i]=T[i]@(em[i+1]*back[i+1]);back[i]/=back[i].sum()
    initial=np.array([1-rho,rho])*em[0]*back[0];initial/=initial.sum();postT=T*(em[1:]*back[1:])[:,None,:];postT/=postT.sum(2,keepdims=True)
    return q,postT,initial
def correlated_selection(p,t,v):
    q,T,initial=posterior(p,t,v);n=len(p);primary=q[:,:,1];secondary=q[:,:,2]
    def forward(prob):
        z=np.zeros((n,5,2));z[0,0]=initial
        for i in range(n-1):
            after=z[i]*(1-prob[i]);after[1:]+=z[i,:-1]*prob[i];z[i+1]=after@T[i]
        return z
    fp=forward(primary);fa=forward(primary+secondary);back=np.zeros((n,5,2));back[-1,0]=1
    for i in range(n-2,-1,-1):
        future=back[i+1]*(1-primary[i+1]);future[1:]+=back[i+1,:-1]*primary[i+1];back[i]=future@T[i].T
    out=np.array([np.sum(fp[i]*primary[i])+sum(np.sum(fa[i,k]*secondary[i]*back[i,:5-k].sum(0)) for k in range(5)) for i in range(n)])
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<1+1e-8 and out.sum()<5+1e-8
    return out
def check():
    rng=np.random.default_rng(1010);p=rng.dirichlet([3,1,1],size=5);t=np.arange(5)*50.;v=[.35,.7,150];rho=v[0];W=weights(p,*v[:2]);T=transitions(t,rho,v[2]);chosen=np.zeros(5);mass=0.;observed=0.;e=(1,3,0)
    for states in itertools.product(range(2),repeat=5):
        prior=[1-rho,rho][states[0]]*np.prod([T[i,states[i],states[i+1]] for i in range(4)])
        for cats in itertools.product(range(3),repeat=5):
            prob=prior*np.prod(W[np.arange(5),states,cats]);ids=tuple(([i for i,k in enumerate(cats) if k==1]+[i for i,k in enumerate(cats) if k==2])[:5]);mass+=prob;chosen[list(ids)]+=prob
            if ids==e:observed+=prob
    inferred=correlated_selection(p,t,v);err=float(abs(inferred-chosen/mass).max());loss=observed_loss([{'p':p,'t':t,'template':template(5,np.array(e)),'den':3}],v)-.01*v[1]**2;errll=abs(loss+np.log(observed/mass)/3);ind=abs(correlated_selection(p,t,[.35,0,150])-inclusion(p[:,1],p[:,2])).max();assert err<1e-12 and errll<1e-12 and ind<1e-12
    (ROOT/'activity_checks.json').write_text(json.dumps({'paths':6**5,'selection_max_error':err,'list_loss_error':errll,'independent_limit_error':float(ind)},indent=2))
def main():
    path=ROOT/'activity_config.json'
    if path.exists():assert json.loads(path.read_text())==CONFIG
    else:path.write_text(json.dumps(CONFIG,indent=2))
    check();d=hand_data();parts=[];audit=[];start=time.time()
    for f in range(4):
        q=d.join(pl.read_parquet(f'artifacts/evidence_session9/nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');bags=[]
        for (pid,),g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');a=g.select('cat_primary','cat_secondary').to_numpy();b=g.select('hist_primary','hist_secondary').to_numpy();p=.5*(a/np.maximum(1,a.sum(1))[:,None]+b/np.maximum(1,b.sum(1))[:,None]);p=np.column_stack([np.maximum(1e-8,1-p.sum(1)),p]);p/=p.sum(1,keepdims=True);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];bags.append({'pid':pid,'family':g['behavior_family'][0],'fold':g['fold'][0],'g':g,'p':p,'t':g['time'].to_numpy()*5000,'template':template(len(g),e),'den':len(e)})
        for family in ['directed_transfer','soft_play','coordinated_isolation']:
            tr=[b for b in bags if b['fold']!=f and b['family']==family and len(b['template'])];va=[b for b in bags if b['fold']==f and b['family']==family];null=[.35,0,150];null_loss=observed_loss(tr,null);fit=minimize(lambda v:observed_loss(tr,v),CONFIG['initial'],method='Nelder-Mead',bounds=CONFIG['bounds'],options={'maxiter':CONFIG['maxiter'],'xatol':.001,'fatol':1e-6});v=fit.x.tolist() if fit.fun<null_loss else null;record={'fold':f,'family':family,'parameters':v,'training_null_loss':null_loss,'training_fit_loss':float(fit.fun),'optimizer_success':bool(fit.success),'evaluations':fit.nfev,'training_pairs':len(tr)};audit.append(record)
            for b in va:
                g=b['g'];s=correlated_selection(b['p'],b['t'],v);new=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*s;parts.append(g.select('pair_id','hand_id','fold','evidence','r29').with_columns(pl.Series('activity',new),pl.Series('activity_inclusion',s)))
            print('activity',f,family,record,'seconds',round(time.time()-start,1),flush=True)
            (ROOT/'activity_audit.json').write_text(json.dumps(audit,indent=2))
    pl.concat(parts).write_parquet(ROOT/'activity_oof.parquet')
if __name__=='__main__':main()
