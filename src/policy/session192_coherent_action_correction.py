\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,time,hashlib,sys
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from sklearn.tree import DecisionTreeRegressor
from scipy.special import softmax
import session11_list_boost as b
from session11_conditional_family import conditional_nll,features
from session8_count_conditioning import conditioned
import session162_action_em_math as math
from session189_pair_event_prototypes import data

ROOT=Path('artifacts/evidence_session192_coherent_action_correction');C=pl.col
CONFIG=dict(b.CONFIG,input_root='artifacts/evidence_session55_current_nested',method=__doc__,
    objective='exact conditional ordered-list NLL/truth count + pair-balanced mean action residual square',
    inference='conditional capped inclusion from coherent action marginals; fixed equal pressure59 mixture',
    bounds='1.5 logit units per action, not per hand')
torch.set_num_threads(2)

def aggregate_logits(logits,bag,nh):
    lp=torch.log_softmax(logits.to(torch.float64),dim=1)
    zero=torch.zeros(nh,dtype=lp.dtype)
    ln0=zero.index_add(0,bag,lp[:,0])
    lnnp=zero.index_add(0,bag,torch.logaddexp(lp[:,0],lp[:,2]))
    a=-torch.expm1(lnnp)
    bprob=torch.exp(lnnp)*(-torch.expm1(torch.clamp(ln0-lnnp,max=0)))
    return torch.stack([torch.log(a.clamp_min(1e-15))-ln0,torch.log(bprob.clamp_min(1e-15))-ln0],dim=1)

def check_math():
    rng=np.random.default_rng(192);p=rng.dirichlet([3,1,1],12);bag=np.repeat(np.arange(4),3)
    z=torch.tensor(np.log(p),requires_grad=True);idx=torch.tensor(bag)
    got=aggregate_logits(z,idx,4);expected=math.hand_probabilities(p,bag,4)[0]
    hp=torch.softmax(torch.column_stack([torch.zeros(4),got]),1).detach().numpy()
    np.testing.assert_allclose(hp,expected,atol=1e-12,rtol=0)
    weights=torch.tensor(rng.normal(size=(4,2)));v=(got*weights).sum();v.backward();grad=z.grad.numpy();err=0.
    for i,k in [(0,0),(1,1),(5,2),(10,1)]:
        zz=z.detach().clone();zz[i,k]+=1e-5;up=(aggregate_logits(zz,idx,4)*weights).sum().item()
        zz[i,k]-=2e-5;down=(aggregate_logits(zz,idx,4)*weights).sum().item();err=max(err,abs((up-down)/2e-5-grad[i,k]))
    assert err<1e-7
    np.testing.assert_allclose(aggregate_logits(z.detach().flip(0),idx.flip(0),4),got.detach(),atol=1e-12,rtol=0)
    return {'hand_marginal_error':float(abs(hp-expected).max()),'finite_difference_gradient_error':err,'row_reversal_exact_within_1e12':True}

def main():
    ROOT.mkdir(exist_ok=True);proof=check_math();(ROOT/'math_verification.json').write_text(json.dumps(proof,indent=2))
    (ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=data();start=time.time()
    native=pl.read_parquet('artifacts/evidence_session122_family_blind_hands/hands.parquet')
    meta=pl.read_parquet('artifacts/evidence_session163_generic_action_data/actions.parquet')
    mapping=native.select('hand_index','pair_id','hand_id').join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='1:1').sort('hand_index')['row'].to_numpy()
    bag=mapping[meta['hand_index'].to_numpy()];idx=torch.tensor(bag.astype(np.int64));raw=np.load('artifacts/evidence_session163_generic_action_data/x.npy',mmap_mode='r')
    b.CONFIG['input_root']=CONFIG['input_root'];b.features=features
    outputs=[];audit=[]
    for f in range(4):
        groups,compact,_,P,M,T,V,D,fv,bid,pos=b.pack(d,f,['relative_time'])
        assert pl.concat([g.select('pair_id','hand_id') for g in groups]).equals(d.select('pair_id','hand_id'))
        ti=torch.tensor(np.flatnonzero((fv!=f)&V.any(1).numpy()));valid=np.flatnonzero(fv==f)
        hand_train=np.isin(bid,ti.numpy());at=hand_train[bag];av=np.isin(bid,valid)[bag]
        fam=np.array([g['behavior_family'][0] for g in groups]);minimums={s:int(D.numpy()[ti.numpy()[fam[ti.numpy()]==s]].min()) for s in set(fam)}
        K=torch.tensor([minimums[s] for s in fam]);hp=softmax(np.column_stack([np.zeros(len(d)),P.numpy()[bid,pos]]),axis=1)
        ap=math.initialize(hp,bag);initial_error=float(abs(math.hand_probabilities(ap,bag,len(d))[0]-hp).max());assert initial_error<1e-10
        prior=torch.tensor(np.log(np.maximum(ap,1e-15)),dtype=torch.float64)
        X=np.nan_to_num(np.column_stack([raw,compact[bag]]),nan=0,posinf=1e6,neginf=-1e6).astype(np.float32)
        pair_index=bid[bag];pair_counts=np.bincount(pair_index,minlength=len(groups));regweight=torch.tensor(1/pair_counts[pair_index],dtype=torch.float64)
        trainmask=torch.tensor(at,dtype=torch.float64);den=D.numpy()[pair_index]
        def objective(A,templates=T,denominators=D):
            z=aggregate_logits(prior+torch.column_stack([torch.zeros(len(A)),A]),idx,len(d))
            padded=torch.zeros((*P.shape[:2],2),dtype=z.dtype);padded[bid,pos]=z
            loss=conditional_nll(padded[ti],templates[ti],V[ti],denominators[ti],M[ti],K[ti]).sum()
            return loss+CONFIG['ridge']*(A.square().sum(1)*regweight*trainmask).sum()
        A=torch.zeros((len(raw),2),requires_grad=True);probe=objective(A);probe.backward();assert not torch.count_nonzero(A.grad[av])
        changed=T.clone();changed[valid]=1;dd=D.clone();dd[valid]=999
        assert float(probe.detach())==float(objective(A.detach(),changed,dd));A=A.detach();initial=float(probe.detach());history=[]
        model={'steps':[],'fold':f,'config':CONFIG,'minimums':minimums,'feature_count':X.shape[1]}
        print('START',f,'actions',len(raw),'features',X.shape[1],'initial',initial,flush=True)
        for it in range(CONFIG['iterations']):
            A.requires_grad_(True);loss=objective(A);loss.backward();grad=A.grad.detach().numpy();a=A.detach().numpy()
            probs=torch.softmax(prior+torch.column_stack([torch.zeros(len(A)),A.detach()]),1).numpy()[:,1:]
            h=np.maximum(probs*(1-probs),.02)/den[:,None]+2*CONFIG['ridge']/pair_counts[pair_index,None]
            target=np.clip(-grad/h,-5,5)
            tree=DecisionTreeRegressor(max_leaf_nodes=10,min_samples_leaf=80,max_features=.7,random_state=1111+f*1000+it)
            tree.fit(X[at],target[at],sample_weight=h[at].mean(1));update=tree.predict(X).astype(np.float32)
            old=float(loss.detach());A=A.detach();step=.15;accepted=False
            with torch.no_grad():
                for _ in range(7):
                    candidate=torch.tensor(np.clip(a+step*update,-1.5,1.5));new=float(objective(candidate))
                    if new<=old+1e-6:accepted=True;break
                    step*=.5
            if not accepted:step=0.;candidate=A;new=old
            A=candidate;model['steps'].append((tree,step));history.append(new)
            if (it+1)%30==0:print('ITER',f,it+1,round(new,4),round(time.time()-start,1),flush=True)
        path=ROOT/f'fold{f}.joblib';joblib.dump(model,path,compress=3);reloaded=joblib.load(path)
        ix=np.flatnonzero(av);replay=b.infer(reloaded,X[ix]);np.testing.assert_allclose(replay,A.numpy()[ix],atol=1e-6,rtol=0)
        np.testing.assert_array_equal(b.infer(reloaded,X[ix[::-1]])[::-1],replay)
        logits=aggregate_logits(prior+torch.column_stack([torch.zeros(len(A)),A]),idx,len(d)).detach().numpy()
        hp=softmax(np.column_stack([np.zeros(len(d)),logits]),axis=1)
        for i in valid:
            g=groups[i];ix=g['row'].to_numpy();inc=conditioned(hp[ix,1:],minimums[g['behavior_family'][0]])
            score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
            outputs.append(g.select('pair_id','hand_id').with_columns(pl.Series('coherent_action',score)))
        audit.append({'fold':f,'training_actions':int(at.sum()),'heldout_actions':int(av.sum()),'initial_marginal_error':initial_error,'initial_loss':initial,'training_loss':history,'final_loss':history[-1],
                      'heldout_gradient_zero':True,'heldout_truth_mutation_invariant':True,'checkpoint_replay':True,'action_row_reversal_exact':True,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));pl.concat(outputs).write_parquet(ROOT/'partial_oof.parquet')
        print('FOLD',f,round(time.time()-start,1),flush=True)
    q=pl.concat(outputs);q.write_parquet(ROOT/'oof.parquet');compare(d,q)

def compare(d,q):
    import session190_action_list_correction as r
    baseline=pl.read_parquet('artifacts/evidence_session190_action_list_correction/oof.parquet').select('pair_id','hand_id','compact_control')
    z=q.rename({'coherent_action':'action_list'}).join(baseline,on=['pair_id','hand_id'],validate='1:1')
    r.ROOT=ROOT;r.compare(d,z)
    p=ROOT/'report.json';report=json.loads(p.read_text());report['method']=__doc__;p.write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='check':print(check_math())
    else:main()
