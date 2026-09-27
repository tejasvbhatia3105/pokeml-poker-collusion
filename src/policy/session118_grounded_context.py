\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import json,time,hashlib
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch import nn
import session11_conditional_family as base
import session11_list_boost as packer
from session62_grounded_list_boost import grounded
from session8_data import hand_data
from session117_fast_count import log_count_at_least,provenance

ROOT=Path('artifacts/evidence_session118_grounded_context')
CONFIG=dict(base.CONFIG,features='35 compact current nested55 + grounded62 action features',
    kinds=['independent','contextual'],input_root='artifacts/evidence_session55_current_nested',
    padding='All hands retained; batches trim padding only',normalization='Compatible outer-training rows only; std floor .05; clip +/-6',
    selection='Two fixed seeds, 80 epochs, no held-out checkpoint selection')

def prepare():
    ROOT.mkdir(exist_ok=True)
    d=hand_data();x,cols=grounded(d)
    old=np.load('artifacts/evidence_session62_grounded_list_boost/grounded_features.npz')['x']
    assert np.array_equal(x,old) and cols==json.load(open('artifacts/evidence_session62_grounded_list_boost/grounded_columns.json'))
    proof=json.load(open('artifacts/evidence_session55_current_nested/input_verification.json'))
    assert proof['saved_models_replayed']==175 and proof['event_probability_error']==0 and proof['all_training_prediction_and_outer_pool_sets_disjoint']
    (ROOT/'grounded_columns.json').write_text(json.dumps(cols,indent=2))
    return d.with_columns(*[pl.Series(c,x[:,i]) for i,c in enumerate(cols)]),cols

def pack(d,cols,f):
    packer.CONFIG=dict(packer.CONFIG,input_root=CONFIG['input_root'])
    groups,small,big,P,M,T,V,D,fv,bid,pos=packer.pack(d,f,cols)
    xx=np.column_stack([small,big]);X=np.zeros((*M.shape,xx.shape[1]),np.float32);X[bid,pos]=xx
    tr=(fv!=f)&V.any(1).numpy();va=fv==f
    mu=X[tr][M.numpy()[tr]].mean(0);sd=np.maximum(.05,X[tr][M.numpy()[tr]].std(0))
    families=np.array([g['behavior_family'][0] for g in groups]);mins={fam:int(D.numpy()[tr&(families==fam)].min()) for fam in set(families)}
    K=torch.tensor([mins[fam] for fam in families]);XX=torch.tensor(np.clip((X-mu)/sd,-6,6))
    assert np.isfinite(X).all() and not np.any(tr&va) and M.sum().item()==45129
    return groups,(XX,P,M,T,V,D,K),np.flatnonzero(tr),np.flatnonzero(va),mu,sd,mins

def batches(tensors,indices):
    ix=torch.tensor(indices,dtype=torch.long,device=tensors[0].device)
    X,P,M,T,V,D,K=[v[ix] for v in tensors];n=int(M.sum(1).max().item())
    return X[:,:n],P[:,:n],M[:,:n],T[:,:,:n],V,D,K

def loss_fn(delta,P,M,T,V,D,K):
    return (base.list_nll(P+delta,T,V,D)+log_count_at_least(P+delta,M,K)/D).mean()+CONFIG['residual_l2']*delta.square().sum()/M.sum()/2

def predict(model,tensors,valid):
    model.eval();out=[]
    with torch.no_grad():
        for i in valid:
            X,P,M,T,V,D,K=batches(tensors,[i]);z=P+model(X,M)
            out.append(torch.softmax(torch.cat([torch.zeros_like(z[:,:,:1]),z],2),2)[0].cpu().numpy())
    return out

def main():
    device=os.environ.get('LIST_DEVICE','mps');assert device!='mps' or torch.backends.mps.is_available()
    d,cols=prepare();base.ROOT=ROOT;base.check()
    prov=provenance();prov[__file__]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config=dict(CONFIG,device=device,provenance=prov)
    cp=ROOT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:cp.write_text(json.dumps(config,indent=2))
    parts=[];audits=[];start=time.time()
    for f in range(4):
        groups,raw,train,valid,mu,sd,mins=pack(d,cols,f);tensors=tuple(x.to(device) for x in raw)
        predictions={}
        for kind in CONFIG['kinds']:
            seedpred=[]
            for seed in CONFIG['seeds']:
                torch.manual_seed(seed+f);rng=np.random.default_rng(seed+f)
                m=base.Model(len(mu),kind).to(device);path=ROOT/f'{kind}_fold{f}_seed{seed}.pt'
                if path.exists():
                    ck=torch.load(path,map_location=device,weights_only=False);assert ck['config']==config
                    m.load_state_dict(ck['state_dict'])
                else:
                    opt=torch.optim.AdamW(m.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
                    m.eval();X,P,M,T,V,D,K=batches(tensors,train[:2])
                    with torch.no_grad():assert torch.count_nonzero(m(X,M)).item()==0
                    history=[]
                    for epoch in range(CONFIG['epochs']):
                        m.train();order=rng.permutation(train);total=0.;count=0
                        for j in range(0,len(order),CONFIG['batch_size']):
                            ix=order[j:j+CONFIG['batch_size']];X,P,M,T,V,D,K=batches(tensors,ix)
                            delta=m(X,M);loss=loss_fn(delta,P,M,T,V,D,K)
                            assert torch.isfinite(loss).item();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                            total+=float(loss.detach().cpu())*len(ix);count+=len(ix)
                        history.append(total/count)
                        if (epoch+1)%20==0:print(f,kind,seed,epoch+1,history[-1],round(time.time()-start,1),flush=True)
                    ck=dict(state_dict={k:v.cpu() for k,v in m.state_dict().items()},mu=mu,sd=sd,minimums=mins,config=config,
                        train_pair_ids=[groups[i]['pair_id'][0] for i in train],valid_pair_ids=[groups[i]['pair_id'][0] for i in valid],history=history)
                    torch.save(ck,path)
                pr=predict(m,tensors,valid);seedpred.append(pr)
                flat=np.concatenate(pr);np.savez_compressed(ROOT/f'{kind}_fold{f}_seed{seed}_pred.npz',probability=flat)
                audits.append(dict(fold=f,kind=kind,seed=seed,train_pairs=len(train),valid_pairs=len(valid),final_train_loss=ck['history'][-1]))
            predictions[kind]=seedpred
        for j,i in enumerate(valid):
            g=groups[i];z=g.select('pair_id','hand_id','fold','evidence')
            for kind,prs in predictions.items():
                inc=np.mean([base.conditioned(p[j][:,1:],mins[g['behavior_family'][0]]) for p in prs],axis=0)
                score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
                z=z.with_columns(pl.Series(kind,score),pl.Series(kind+'_inclusion',inc))
            parts.append(z)
        pl.concat(parts).write_parquet(ROOT/'partial_oof.parquet')
    pl.concat(parts).write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audits,indent=2))
    print('complete',time.time()-start,flush=True)

if __name__=='__main__':main()
