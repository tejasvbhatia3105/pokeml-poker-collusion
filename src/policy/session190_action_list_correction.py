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
import json,time,hashlib
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from sklearn.tree import DecisionTreeRegressor
import session11_list_boost as b
from session11_conditional_family import conditional_nll,features
from session8_count_conditioning import conditioned
from session189_pair_event_prototypes import data,action_means,FIELDS

ROOT=Path('artifacts/evidence_session190_action_list_correction');C=pl.col
CONFIG=dict(b.CONFIG,input_root='artifacts/evidence_session55_current_nested',method=__doc__,objective='exact conditional ordered-list NLL / truth count plus mean residual square')
torch.set_num_threads(2)

def objective(P,A,M,T,V,D,train,K):
    z=conditional_nll(P[train]+A[train],T[train],V[train],D[train],M[train],K[train]).sum()
    reg=(A[train].square().sum(2)*M[train]).sum(1)/M[train].sum(1)
    return z+CONFIG['ridge']*reg.sum()

def main():
    ROOT.mkdir(exist_ok=True);start=time.time();d=data();raw,count=action_means(d)
    raw=np.column_stack([(np.sign(raw)*np.log1p(abs(raw))).reshape(len(d),-1),np.log1p(count)]).astype(np.float32)
    cols=[f'raw_{k}_{c}' for k in range(4) for c in FIELDS]+[f'action_count_{k}' for k in range(4)]
    assert len(cols)==raw.shape[1]==84
    d=d.with_columns(*[pl.Series(c,raw[:,i]) for i,c in enumerate(cols)])
    b.CONFIG['input_root']=CONFIG['input_root'];b.features=features
    (ROOT/'config.json').write_text(json.dumps(dict(CONFIG,columns=cols),indent=2));outputs=[];audit=[];checks=[];archived_audit=json.loads((ROOT/'audit.json').read_text()) if (ROOT/'audit.json').exists() else []
    for f in range(4):
        groups,compact,extra,P,M,T,V,D,fv,bid,pos=b.pack(d,f,cols)
        train=np.flatnonzero((fv!=f)&V.any(1).numpy());valid=np.flatnonzero(fv==f)
        rt=np.isin(bid,train);rv=np.isin(bid,valid);ti=torch.tensor(train)
        fam=np.array([g['behavior_family'][0] for g in groups]);minimums={s:int(D.numpy()[train[fam[train]==s]].min()) for s in set(fam)}
        K=torch.tensor([minimums[s] for s in fam]);rowweight=1/D.numpy()[bid]
                                                                                
        probe=torch.zeros_like(P,requires_grad=True);loss=objective(P,probe,M,T,V,D,ti,K);loss.backward()
        assert not torch.count_nonzero(probe.grad[valid]);tt=T.clone();tt[valid]=1;dd=D.clone();dd[valid]=999
        assert float(loss.detach())==float(objective(P,probe.detach(),M,tt,V,dd,ti,K))
                                                                                   
        control_path=Path(f'artifacts/evidence_session62_grounded_list_boost/list_boost_compact_fold{f}.joblib')
        control_logits=None
        if not (control_path.stat().st_flags & 0x40000000):
            control=joblib.load(control_path)
            ca=b.infer(control,compact);control_logits=P.numpy().copy();control_logits[bid,pos]+=ca
        print('START',f,'compact_checkpoint_local',control_logits is not None,flush=True)
        predicted={};X=np.nan_to_num(np.column_stack([compact,extra]),nan=0,posinf=1e6,neginf=-1e6)
        model={'steps':[],'fold':f,'config':CONFIG,'minimums':minimums,'columns':cols,'feature_count':X.shape[1]}
        A=torch.zeros_like(P);initial=float(objective(P,A,M,T,V,D,ti,K));history=[]
        existing=ROOT/f'fold{f}.joblib'
        if existing.exists():
            model=joblib.load(existing)
            assert model['config']==CONFIG
            aa=b.infer(model,X);A=torch.zeros_like(P);A[bid,pos]=torch.tensor(aa)
            prior_audit=archived_audit
            history=next(r['training_loss'] for r in prior_audit if r['fold']==f)
            assert abs(float(objective(P,A,M,T,V,D,ti,K))-history[-1])<1e-4
            print('RESUMED',f,flush=True)
        else:
            for it in range(CONFIG['iterations']):
                A.requires_grad_(True);loss=objective(P,A,M,T,V,D,ti,K);loss.backward()
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
                        new=float(objective(P,candidate,M,T,V,D,ti,K))
                        if new<=old+1e-6:accepted=True;break
                        step*=.5
                if not accepted:step=0.;candidate=A;new=old
                A=candidate;model['steps'].append((tree,step));history.append(new)
                if (it+1)%30==0:print('ITER',f,it+1,round(new,4),round(time.time()-start,1),flush=True)
        path=ROOT/f'fold{f}.joblib';joblib.dump(model,path,compress=3);reload=joblib.load(path)
        np.testing.assert_allclose(b.infer(reload,X[rv]),A.numpy()[bid[rv],pos[rv]],atol=1e-6,rtol=0)
        old_oof=pl.read_parquet('artifacts/evidence_session62_grounded_list_boost/conditional_boost_oof.parquet')
        error=0.
        for i in valid:
            g=groups[i];n=len(g);scores={}
            expected=g.select('pair_id','hand_id').join(old_oof.select('pair_id','hand_id','compact'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['compact'].to_numpy()
            arms=[('action_list',(P+A)[i,:n].numpy())]
            if control_logits is not None:arms.append(('compact_control',control_logits[i,:n]))
            else:scores['compact_control']=expected
            for name,z in arms:
                prob=torch.softmax(torch.tensor(np.column_stack([np.zeros(n,np.float32),z])),1).numpy()
                inc=conditioned(prob[:,1:],minimums[g['behavior_family'][0]])
                scores[name]=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
            error=max(error,float(abs(expected-scores['compact_control']).max()))
            outputs.append(g.select('pair_id','hand_id').with_columns(*[pl.Series(k,scores[k]) for k in ['compact_control','action_list']]))
        assert error<1e-7,error
        audit.append({'fold':f,'train_pairs':len(train),'validation_pairs':len(valid),'initial_loss':initial,'final_loss':history[-1],'training_loss':history,'checkpoint_replay':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        checks.append({'fold':f,'compact62_checkpoint_replayed':control_logits is not None,'compact62_max_error':error if control_logits is not None else None,'heldout_template_and_count_invariant':True,'heldout_gradient_zero':True})
        (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'verification.json').write_text(json.dumps(checks,indent=2))
        pl.concat(outputs).write_parquet(ROOT/'partial_oof.parquet')
        print('FOLD',f,round(time.time()-start,1),flush=True)
    q=pl.concat(outputs);q.write_parquet(ROOT/'oof.parquet');compare(d,q)

def compare(d,q):
    from session189_pair_event_prototypes import score_report
    import session189_pair_event_prototypes as report_module
    report_module.ROOT=ROOT
    z=d.select('pair_id','hand_id').join(q,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    current=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    risk=d.select('pair_id').join(pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score'),on='pair_id',validate='m:1',maintain_order='left')['risk_score'].to_numpy()
    old=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['conditional_family'].to_numpy()
    pressure=current['pressure59'].to_numpy()
    pred={'r33':current['equal'].to_numpy(),'raw':z['compact_control'].to_numpy(),'prototype':z['action_list'].to_numpy(),
          'raw_transport':.5*(pressure+z['compact_control'].to_numpy()),'prototype_transport':.5*(pressure+z['action_list'].to_numpy())}
    pred={k:np.where(risk<.05,old,p) for k,p in pred.items()}
    r=score_report(d,pred)
                                                                
    names={'raw':'compact_control','prototype':'action_list','raw_transport':'compact_pressure','prototype_transport':'action_pressure'}
    r['results']={names.get(k,k):v for k,v in r['results'].items()};r['action_vs_compact']={names[k]:v for k,v in r.pop('matched_prototype_vs_raw').items()};r['method']=__doc__
    (ROOT/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
