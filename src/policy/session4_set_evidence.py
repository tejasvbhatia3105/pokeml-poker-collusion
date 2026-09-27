\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import itertools,json,time
import numpy as np
import polars as pl
import torch
from scipy.special import expit,logit
torch.set_num_threads(4)
ROOT=Path(os.environ.get('EVIDENCE_SET_ROOT','artifacts/evidence_session4'));K=12
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
UNAMES=['base_logit','relative_time','shared_position','candidate_position','before_mass','after_mass','left_gap','right_gap']
PNAMES=['shared_gap1','shared_gap3','shared_gap10','shared_gap30','table_gap25','table_gap100','table_gap400','same_payoff','opposite_payoff','signature_cos','signature_l1','near_signature','near_same_payoff','same_strong_action']
REG=.1
S={}
for k in range(K+1):
    z=np.zeros((len(list(itertools.combinations(range(K),k))),K),dtype=np.float32)
    for i,chosen in enumerate(itertools.combinations(range(K),k)):z[i,list(chosen)]=1
    S[k]=z
I,J=np.triu_indices(K,1)
def design(g):
    g=g.sort(['time','hand_id']);n=len(g);scores=g['base_score'].to_numpy()
                                                                               
    idx=np.lexsort((g['hand_id'].to_numpy(),-scores))[:K]
    assert len(idx)==K
    z=g[idx];sc=scores[idx];t=z['time'].to_numpy()*5000
    tc=np.argsort(np.argsort(t));sg=np.abs(idx[:,None]-idx[None,:]);tg=np.abs(t[:,None]-t[None,:])
    left=np.full(K,1.);right=np.full(K,1.)
    order=np.argsort(t);gaps=np.diff(t[order])/3000
    left[order[1:]]=gaps;right[order[:-1]]=gaps
    mass=np.cumsum(scores)
    U=np.stack([logit(np.clip(sc,1e-5,1-1e-5)),t/3000,idx/n,tc/(K-1),
                np.log1p(mass[idx]-sc),np.log1p(mass[-1]-mass[idx]),left,right],1)
    direction=np.sign(z['net_direction'].to_numpy());pay=direction[:,None]*direction[None,:]
    rc=[c for c in z.columns if c.endswith('_r')];a=z.select(rc).to_numpy()
    v=z.select([c[:-2]+'_v' for c in rc]).to_numpy();a=a/np.sqrt(1+v)
    a=np.sign(a)*np.log1p(np.abs(a));norm=np.linalg.norm(a,axis=1);unit=a/(norm[:,None]+1e-8)
    cos=unit@unit.T;l1=np.exp(-np.mean(np.abs(a[:,None,:]-a[None,:,:]),axis=2))
    same=np.argmax(np.abs(a),axis=1)[:,None]==np.argmax(np.abs(a),axis=1)[None,:]
    near=np.exp(-sg/10)
    P=np.stack([np.exp(-sg/q) for q in [1,3,10,30]]+
               [np.exp(-tg/q) for q in [25,100,400]]+
               [(pay>0).astype(float),(pay<0).astype(float),cos,l1,near*cos,near*(pay>0),same.astype(float)],2)
    y=z['evidence'].to_numpy().astype(int)
    return dict(pid=z['pair_id'][0],fold=int(z['fold'][0]),family=z['behavior_family'][0],
                hand=z['hand_id'].to_list(),y=y,den=min(5,int(g['evidence'].sum())),U=U.astype('float32'),P=P[I,J].astype('float32'))
def features(b,k,pairwise):
    s=S[k];u=s@b['U']
    q=(s[:,I]*s[:,J])@b['P']/max(1,k-1)
    x=np.concatenate([u,q],1) if pairwise else u
                                                     
    delta=np.zeros((len(x),x.shape[1]*3),dtype='float32');f=FAMILIES.index(b['family'])
    delta[:,f*x.shape[1]:(f+1)*x.shape[1]]=x
    return np.concatenate([x,delta],1)
def fit(bags,pairwise):
    xs=[];ys=[]
    for b in bags:
        k=int(b['y'].sum());x=features(b,k,pairwise)
        if k==0:continue
        truth=int(np.flatnonzero(np.all(S[k]==b['y'],axis=1))[0]);xs.append(x);ys.append(x[truth])
    widths=[len(x) for x in xs];maxw=max(widths);dim=xs[0].shape[1]
    xx=np.zeros((len(xs),maxw,dim),dtype='float32');mask=np.zeros((len(xs),maxw),bool)
    for i,x in enumerate(xs):xx[i,:len(x)]=x;mask[i,:len(x)]=True
    xx=torch.tensor(xx);truth=torch.tensor(np.stack(ys));mask=torch.tensor(mask)
    initial=torch.zeros(dim);initial[0]=1.
    w=torch.nn.Parameter(initial.clone());opt=torch.optim.LBFGS([w],lr=1,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();energy=(xx@w).masked_fill(~mask,-1e9)
        loss=(torch.logsumexp(energy,1)-truth@w).mean()+REG*((w-initial)**2).sum()
        loss.backward();return loss
    opt.step(closure)
    return w.detach().numpy()
def predict(b,w,pairwise):
                                                                   
    x=features(b,5,pairwise);v=x@w;v=np.exp(v-v.max());v/=v.sum();s=S[5]
    joint=(s.T*v)@s;marg=np.diag(joint);chosen=[]
    for _ in range(5):
        value=marg.copy()
        if chosen:value+=joint[:,chosen].sum(1)
        value[chosen]=-np.inf;chosen.append(int(np.argmax(value)))
    return chosen
def ap(b,chosen):
    y=b['y'][chosen];return float(np.sum(y*np.cumsum(y)/np.arange(1,len(y)+1))/b['den'])
def main():
    d=pl.read_parquet('artifacts/policy/evidence_training.parquet')
    index=pl.read_parquet(ROOT/'hand_index.parquet').select('pair_id','hand_id','fold')
    d=d.join(index,on=['pair_id','hand_id']);rows=[];t=time.time()
    for f in range(4):
        p=ROOT/f'nested_outer{f}.parquet'
        if not p.exists():raise RuntimeError(f'Finish nested base generation first: {p}')
        q=d.join(pl.read_parquet(p),on=['pair_id','hand_id'])
        bags=[design(g) for _,g in q.group_by('pair_id',maintain_order=True)]
        tr=[b for b in bags if b['fold']!=f];va=[b for b in bags if b['fold']==f]
        preds={}
        for mode in ['unary','set']:
            w=fit(tr,mode=='set');np.save(ROOT/f'{mode}_weights_fold{f}.npy',w)
            preds[mode]=[predict(b,w,mode=='set') for b in va]
        for i,b in enumerate(va):
            row=dict(pair_id=b['pid'],fold=f,family=b['family'],base=ap(b,list(range(5))),
                     oracle12=float(min(5,b['y'].sum())/b['den']))
            for mode in preds:row[mode]=ap(b,preds[mode][i])
            rows.append(row)
        print('set fold',f,'elapsed',round(time.time()-t,1),flush=True)
    res=pl.DataFrame(rows);res.write_csv(ROOT/'set_validation.csv')
    print(res.group_by('family').agg(pl.col('base','unary','set','oracle12').mean()))
    print('overall',res.select(pl.col('base','unary','set','oracle12').mean()).to_dicts(),flush=True)
    rng=np.random.default_rng(410);labs=pl.read_parquet(ROOT/'hand_index.parquet').select('pair_id','table_id').unique()
    res=res.join(labs,on='pair_id');pool=res.group_by('table_id').agg(pl.col('base','unary','set').sum(),pl.len().alias('n'))
    ix=rng.integers(0,len(pool),size=(3000,len(pool)));n=pool['n'].to_numpy()[ix].sum(1);stats={}
    for mode in ['unary','set']:
        delta=pool[mode].to_numpy()-pool['base'].to_numpy();boot=delta[ix].sum(1)/n
        stats[mode]={'gain':res[mode].mean()-res['base'].mean(),'ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist()}
    (ROOT/'set_results.json').write_text(json.dumps({'config':{'K':K,'reg':REG,'unary_features':UNAMES,'pair_features':PNAMES},'results':stats},indent=2))
if __name__=='__main__':main()
