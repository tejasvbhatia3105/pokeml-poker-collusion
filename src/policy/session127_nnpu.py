\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

ROOT = Path('artifacts/pair_session127_nnpu')
WINDOWS = ['full', 'first_2000', 'last_2000']
CFG = dict(steps=600, batch=16384, hidden=[64,32], lr=.003,
           weight_decay=.01, min_train_hands=15, window_weight=.5,
           seed=12700, prior_multipliers=[2,4], beta=0, gamma=1,
           loss='logistic', inputs='190 ordinary-policy residuals',
           population='all eligible training views, including labelled examples',
           positive='trusted positive with >=1 listed evidence hand in view',
           prior='training weighted labelled-positive density times multiplier',
           caveat='Unknown true prior and selected positive sample; SCAR is unverified.')

def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def load():
    cols=json.loads(Path('artifacts/policy/residual_columns.json').read_text())
    folds=json.loads(Path('artifacts/folds.json').read_text())
    tf={t:f['fold'] for f in folds for t in f['valid_tables']}
    lab=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family')
    xs=[]; ms=[]
    for wi,w in enumerate(WINDOWS):
        folder=Path('artifacts/policy/pair_features') if wi==0 else Path(f'artifacts/policy/full_window_stress/{w}/pair_features')
        d=pl.concat([pl.read_parquet(p).filter(pl.col('phase')=='development') for p in sorted(folder.glob('*.parquet'))]).filter(pl.col('table_id').is_in(list(tf))).sort('pair_id')
        truth=pl.read_parquet(f'artifacts/evidence_session91_family_uncertainty/{w}.parquet',columns=['pair_id','n_truth'])
        m=d.select('pair_id','table_id',pl.col('policy_n_hands').alias('n_hands')).join(lab,on='pair_id',how='left').join(truth,on='pair_id',how='left').with_columns(pl.col('label').fill_null(-1),pl.col('n_truth').fill_null(0),pl.lit(w).alias('window'))
        assert m['pair_id'].to_list()==d['pair_id'].to_list(), 'Feature/metadata join reordered rows'
        m=m.with_columns(pl.Series('fold',[tf[t] for t in m['table_id']]))
        x=d.select(cols).to_numpy().astype(np.float32)
        assert np.isfinite(x).all()
        np.sign(x,out=(sg:=np.empty_like(x))); np.abs(x,out=x); np.log1p(x,out=x); x*=sg
        xs.append(x);ms.append(m)
    x=np.concatenate(xs);m=pl.concat(ms).with_row_index('row')
    assert m.select('pair_id','window').unique().height==m.height
    return x,m,cols

class Net(nn.Module):
    def __init__(self,n):
        super().__init__();self.layers=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):return self.layers(x).squeeze(-1)

def terms(p,u,pw,prior):
    pos=prior*(F.softplus(-p)*pw).sum()
    neg=F.softplus(u).mean()-prior*(F.softplus(p)*pw).sum()
    return pos,neg

def math_check():
    p=torch.tensor([-.8,1.2],dtype=torch.float64,requires_grad=True)
    n=torch.tensor([-2.,.4,1.],dtype=torch.float64,requires_grad=True)
    prior=.2
    ru=prior*F.softplus(p).mean()+(1-prior)*F.softplus(n).mean()
    pu=prior*F.softplus(-p).mean()+ru-prior*F.softplus(p).mean()
    pn=prior*F.softplus(-p).mean()+(1-prior)*F.softplus(n).mean()
    a=torch.autograd.grad(pu,(p,n),retain_graph=True);b=torch.autograd.grad(pn,(p,n))
    err=max(abs(float(pu.detach()-pn.detach())),*(float((i-j).abs().max()) for i,j in zip(a,b)))
    assert err<1e-14
                                                                     
    z=torch.tensor(1.,requires_grad=True);neg=-z*z;grad=torch.autograd.grad(-neg,z)[0]
    assert -(z.detach()-.001*grad)**2 > neg.detach()
    return dict(mixture_risk_and_gradient_max_error=err,correction_direction=True)

def selection(m,f,labels=None):
    label=m['label'].to_numpy() if labels is None else labels
    tr=(m['fold'].to_numpy()!=f)&(m['n_hands'].to_numpy()>=CFG['min_train_hands'])
    pi=np.flatnonzero(tr&(label==1)&(m['n_truth'].to_numpy()>0))
    ui=np.flatnonzero(tr)
    w=np.where(m['window'].to_numpy()=='full',1.,CFG['window_weight'])
    return ui,pi,w

def predict(model,x,device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+32768]).to(device)).cpu().numpy() for i in range(0,len(x),32768)])

def train():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not list(ROOT.glob('*.pt')), 'Do not overwrite completed fits'
    (ROOT/'config.json').write_text(json.dumps(CFG,indent=2))
    math_check();x,m,cols=load();m.write_parquet(ROOT/'metadata.parquet')
    (ROOT/'features.json').write_text(json.dumps(cols))
    device='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(4)
    print('loaded',x.shape,'device',device,flush=True);t0=time.time(); records=[]
    for f in range(4):
        ui,pi,w=selection(m,f);vi=np.flatnonzero(m['fold'].to_numpy()==f)
                                                                                 
        mu=np.average(x[ui],axis=0,weights=w[ui]).astype(np.float32)
        sd=np.sqrt(np.average((x[ui]-mu)**2,axis=0,weights=w[ui])).astype(np.float32).clip(.001)
        xu=torch.from_numpy(np.clip((x[ui]-mu)/sd,-8,8)).to(device)
        xp=torch.from_numpy(np.clip((x[pi]-mu)/sd,-8,8)).to(device)
        xv=np.clip((x[vi]-mu)/sd,-8,8)
        pw=torch.from_numpy((w[pi]/w[pi].sum()).astype(np.float32)).to(device)
        uw=w[ui]/w[ui].sum(); density=float(w[pi].sum()/w[ui].sum())
        for mult in CFG['prior_multipliers']:
            prior=density*mult
            for arm in ['naive','nnpu']:
                torch.manual_seed(CFG['seed']+f);rng=np.random.default_rng(CFG['seed']+f)
                model=Net(x.shape[1]).to(device)
                with torch.no_grad():
                    model.layers[-1].weight.zero_();model.layers[-1].bias.fill_(np.log(prior/(1-prior)))
                opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
                correction=0; trace=[]
                for step in range(CFG['steps']):
                    idx=rng.choice(len(ui),CFG['batch'],replace=True,p=uw)
                    up=model(xu[torch.from_numpy(idx).to(device)]);pp=model(xp)
                    pos,neg=terms(pp,up,pw,prior)
                    if arm=='naive':objective=pos+F.softplus(up).mean()
                    elif float(neg.detach())<0:objective=-neg;correction+=1
                    else:objective=pos+neg
                    opt.zero_grad();(objective/prior).backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    if step%100==0 or step==CFG['steps']-1:
                        trace.append(dict(step=step,pos=float(pos.detach()),neg=float(neg.detach()),objective=float(objective.detach())))
                key=f'{arm}_prior{mult}_fold{f}';pr=predict(model,xv,device)
                state={k:v.cpu() for k,v in model.state_dict().items()}
                rec=dict(key=key,fold=f,arm=arm,multiplier=mult,prior=prior,density=density,corrections=correction,trace=trace,
                         train_rows=len(ui),positive_views=len(pi),train_indices_sha=digest(ui),positive_indices_sha=digest(pi),
                         feature_sha=digest(x),prediction_sha=digest(pr))
                torch.save(dict(state=state,mu=mu,sd=sd,record=rec),ROOT/f'{key}.pt')
                m[vi].select('row','pair_id','window','fold').with_columns(pl.Series('logit',pr)).write_parquet(ROOT/f'{key}.parquet')
                records.append(rec);(ROOT/'training.json').write_text(json.dumps(records,indent=2))
                print(key,'prior',round(prior,6),'corrections',correction,'seconds',round(time.time()-t0),flush=True)
        del xu,xp
        if device=='mps':torch.mps.empty_cache()
    evaluate(m)

def metrics(m,score):
    order=np.lexsort((m['pair_id'].to_numpy(),-score));r=np.empty(len(m),int);r[order]=np.arange(len(m))
    positive=(m['label'].to_numpy()==1)&(m['n_truth'].to_numpy()>0)
    other=~positive
                                                                                    
    others_before=np.cumsum(other[order])-other[order];cnt=np.empty(len(m),int);cnt[order]=others_before
    weak=positive&(m['n_truth'].to_numpy()<=3);trusted=(m['label'].to_numpy()==0)|positive
    out=dict(n_positive=int(positive.sum()),n_weak=int(weak.sum()),trusted_AP=float(average_precision_score(positive[trusted],score[trusted])))
    for k in [100,300,1000]:
        out[f'recall_at_{k}_other']=float(np.mean(cnt[positive]<k))
        out[f'weak_recall_at_{k}_other']=float(np.mean(cnt[weak]<k)) if weak.any() else None
        top=order[:k];out[f'known_negatives_top{k}']=int((m['label'].to_numpy()[top]==0).sum());out[f'unknown_top{k}']=int((m['label'].to_numpy()[top]==-1).sum())
    return out

def evaluate(m):
    result={}
    for mult in CFG['prior_multipliers']:
        for arm in ['naive','nnpu']:
            z=pl.concat([pl.read_parquet(ROOT/f'{arm}_prior{mult}_fold{f}.parquet') for f in range(4)]).sort('row')
            assert np.array_equal(z['row'].to_numpy(),m['row'].to_numpy())
            for w in WINDOWS:
                mask=m['window']==w
                result[f'{arm}_prior{mult}/{w}']=metrics(m.filter(mask),z.filter(mask)['logit'].to_numpy())
    for w in WINDOWS:
        b=pl.read_parquet(f'artifacts/evidence_session91_family_uncertainty/{w}.parquet').select('pair_id','risk_score','gbdt_risk')
        z=m.filter(pl.col('window')==w).join(b,on='pair_id',how='left');assert z['risk_score'].null_count()==0
        for col in ['risk_score','gbdt_risk']:result[f'current_rescored_{col}/{w}']=metrics(z,z[col].to_numpy())
    (ROOT/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def verify():
    x,m,cols=load();torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu';checks=[]
    for path in sorted(ROOT.glob('*.pt')):
        ck=torch.load(path,weights_only=False,map_location='cpu');rec=ck['record'];f=rec['fold'];ui,pi,w=selection(m,f)
        assert digest(x)==rec['feature_sha'] and digest(ui)==rec['train_indices_sha'] and digest(pi)==rec['positive_indices_sha']
        mutated=m['label'].to_numpy().copy();mutated[m['fold'].to_numpy()==f]=1-mutated[m['fold'].to_numpy()==f]
        uj,pj,_=selection(m,f,mutated);assert np.array_equal(ui,uj) and np.array_equal(pi,pj)
        mu=np.average(x[ui],axis=0,weights=w[ui]).astype(np.float32)
        sd=np.sqrt(np.average((x[ui]-mu)**2,axis=0,weights=w[ui])).astype(np.float32).clip(.001)
        assert np.array_equal(mu,ck['mu']) and np.array_equal(sd,ck['sd'])
        vi=np.flatnonzero(m['fold'].to_numpy()==f);xv=np.clip((x[vi]-mu)/sd,-8,8)
        model=Net(len(cols)).to(device);model.load_state_dict(ck['state']);pr=predict(model,xv,device)
        saved=pl.read_parquet(path.with_suffix('.parquet'))['logit'].to_numpy();err=float(abs(pr-saved).max());assert err<1e-4
        reverse=predict(model,xv[::-1].copy(),device)[::-1];rev=float(abs(reverse-pr).max());assert rev<1e-4
        checks.append(dict(key=path.stem,replay_max_error=err,reversal_max_error=rev,fold_label_mutation=True,normalization_recomputed=True))
        print('verified',path.stem,err,flush=True)
    assert len(checks)==16
    (ROOT/'verification.json').write_text(json.dumps(dict(math=math_check(),models=checks),indent=2))

if __name__=='__main__':
    {'train':train,'verify':verify,'math':lambda:print(math_check())}[sys.argv[1]]()
