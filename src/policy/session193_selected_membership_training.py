\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json, time, hashlib
from pathlib import Path
import numpy as np
import polars as pl
import torch, joblib
from sklearn.tree import DecisionTreeRegressor
import session11_list_boost as b
from session11_conditional_family import features
from session193_selection_marginals import ROOT, selected_probabilities
from session189_pair_event_prototypes import data, action_means, FIELDS

C=pl.col
CONFIG=dict(b.CONFIG, input_root='artifacts/evidence_session55_current_nested', method=__doc__,
            objective='Bernoulli NLL of final selected-hand score / truth count + pair-balanced residual penalty',
            matched_control='session190 action-summary exact ordered-list NLL',
            scope='same compatible-list training pairs as190; preserve original low-risk fallback')
torch.set_num_threads(2)

def objective(P,A,M,Y,D,train,K,fixed):
    score=fixed[train]+.5*selected_probabilities(P[train]+A[train],M[train],K[train])
    score=score.clamp(1e-7,1-1e-7);y=Y[train]
    loss=-(y*score.log()+(1-y)*torch.log1p(-score))*M[train]
    reg=(A[train].square().sum(2)*M[train]).sum(1)/M[train].sum(1)
    return (loss.sum(1)/D[train]+CONFIG['ridge']*reg).sum()

def main(design=None, control=None):
    ROOT.mkdir(exist_ok=True);assert (ROOT/'math_verification.json').exists()
    (ROOT/'training_config.json').write_text(json.dumps(CONFIG,indent=2))
    if design is None:
        d=data();raw,count=action_means(d)
        raw=np.column_stack([(np.sign(raw)*np.log1p(abs(raw))).reshape(len(d),-1),np.log1p(count)]).astype(np.float32)
        cols=[f'raw_{k}_{c}' for k in range(4) for c in FIELDS]+[f'action_count_{k}' for k in range(4)]
        d=d.with_columns(*[pl.Series(c,raw[:,i]) for i,c in enumerate(cols)])
    else:
        d,cols=design()
    b.CONFIG['input_root']=CONFIG['input_root'];b.features=features
    outputs=[];audit=[];start=time.time()
    for f in range(4):
        groups,compact,extra,P,M,T,V,D,fv,bid,pos=b.pack(d,f,cols)
        X=np.nan_to_num(np.column_stack([compact,extra]),nan=0,posinf=1e6,neginf=-1e6)
        train=np.flatnonzero((fv!=f)&V.any(1).numpy());valid=np.flatnonzero(fv==f)
        rt=np.isin(bid,train);rv=np.isin(bid,valid);ti=torch.tensor(train)
        fam=np.array([g['behavior_family'][0] for g in groups])
        minimums={s:int(D.numpy()[train[fam[train]==s]].min()) for s in set(fam)}
        K=torch.tensor([minimums[s] for s in fam]);rowweight=1/D.numpy()[bid]
        Y=torch.zeros_like(M,dtype=torch.float64);fixed=torch.zeros_like(Y)
        for i,g in enumerate(groups):
            Y[i,:len(g)]=torch.tensor(g['evidence'].to_numpy().astype(np.float64))
            fixed[i,:len(g)]=torch.tensor(.25*(g['base'].to_numpy()+g['cat_inclusion'].to_numpy()))
        A=torch.zeros_like(P,requires_grad=True);probe=objective(P,A,M,Y,D,ti,K,fixed);probe.backward()
        assert torch.count_nonzero(A.grad[valid])==0
        yy=Y.clone();yy[valid]=1-yy[valid];dd=D.clone();dd[valid]=999
        assert float(probe.detach())==float(objective(P,A.detach(),M,yy,dd,ti,K,fixed))
        A=A.detach();history=[];initial=float(probe.detach())
        model={'steps':[],'fold':f,'config':CONFIG,'minimums':minimums,'columns':cols,'feature_count':X.shape[1]}
        print('START',f,'loss',initial,'features',X.shape[1],flush=True)
        for it in range(CONFIG['iterations']):
            A.requires_grad_(True);loss=objective(P,A,M,Y,D,ti,K,fixed);loss.backward()
            grad=A.grad.detach().numpy()[bid,pos];a=A.detach().numpy()[bid,pos]
            prob=torch.softmax(torch.cat([torch.zeros_like(P[:,:,:1]),P+A.detach()],2),2).numpy()[bid,pos,1:]
            h=np.maximum(prob*(1-prob),.02)*rowweight[:,None]+2*CONFIG['ridge']/M.sum(1).numpy()[bid,None]
            target=np.clip(-grad/h,-5,5)
            tree=DecisionTreeRegressor(max_leaf_nodes=10,min_samples_leaf=80,max_features=.7,random_state=1111+f*1000+it)
            tree.fit(X[rt],target[rt],sample_weight=h[rt].mean(1));update=tree.predict(X).astype(np.float32)
            old=float(loss.detach());A=A.detach();step=.15;accepted=False
            with torch.no_grad():
                for _ in range(7):
                    candidate=torch.zeros_like(P);candidate[bid,pos]=torch.tensor(np.clip(a+step*update,-1.5,1.5))
                    new=float(objective(P,candidate,M,Y,D,ti,K,fixed))
                    if new<=old+1e-6:accepted=True;break
                    step*=.5
                if not accepted:step=0.;candidate=A;new=old
            A=candidate;model['steps'].append((tree,step));history.append(new)
            if (it+1)%30==0:print('ITER',f,it+1,round(new,4),round(time.time()-start,1),flush=True)
        path=ROOT/f'fold{f}.joblib';joblib.dump(model,path,compress=3);reloaded=joblib.load(path)
        replay=b.infer(reloaded,X[rv]);error=float(abs(replay-A.numpy()[bid[rv],pos[rv]]).max())
        assert error<1e-6
        np.testing.assert_array_equal(b.infer(reloaded,X[rv][::-1])[::-1],replay)
        with torch.no_grad():score=fixed+.5*selected_probabilities(P+A,M,K)
        from scipy.special import softmax
        from session8_count_conditioning import conditioned
        reference_error=0.
        for i in valid:
            g=groups[i];s=score[i,:len(g)].numpy()
            p=softmax(np.column_stack([np.zeros(len(g)),(P+A)[i,:len(g)].numpy()]),axis=1)
            other=fixed[i,:len(g)].numpy()+.5*conditioned(p[:,1:],minimums[g['behavior_family'][0]])
            reference_error=max(reference_error,float(abs(s-other).max()))
            outputs.append(g.select('pair_id','hand_id').with_columns(pl.Series('selection_membership',s)))
        assert reference_error<1e-7
        audit.append(dict(fold=f,initial_loss=initial,training_loss=history,final_loss=history[-1],
                          train_pairs=len(train),heldout_pairs=len(valid),heldout_gradient_zero=True,
                          heldout_truth_mutation_invariant=True,checkpoint_replay_error=error,
                          reversed_rows_exact=True,inference_numpy_error=reference_error,
                          model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));pl.concat(outputs).write_parquet(ROOT/'partial_oof.parquet')
        print('FOLD',f,round(time.time()-start,1),flush=True)
    q=pl.concat(outputs);q.write_parquet(ROOT/'oof.parquet')
    import session190_action_list_correction as report
    report.ROOT=ROOT
    old=control() if control is not None else pl.read_parquet('artifacts/evidence_session190_action_list_correction/oof.parquet').select('pair_id','hand_id',C('action_list').alias('compact_control'))
    report.compare(d,q.rename({'selection_membership':'action_list'}).join(old,on=['pair_id','hand_id'],validate='1:1'))
    path=ROOT/'report.json';r=json.loads(path.read_text());r['method']=__doc__
    r['arm_names']={'compact_control':'matched190_joint_list','action_list':'193_selected_membership',
                    'compact_pressure':'matched190_with_pressure59','action_pressure':'193_with_pressure59'}
    path.write_text(json.dumps(r,indent=2))

if __name__=='__main__':
    main()
