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
import json,time
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from sklearn.tree import DecisionTreeRegressor
import session11_list_boost as b
import session190_action_list_correction as old
import session202_jev_event_history as reporting
from session11_conditional_family import features
from session8_count_conditioning import conditioned
from session189_pair_event_prototypes import data
from session197_jev_pilot import QUESTIONS,available

ROOT=Path('artifacts/evidence_session203_jev_exact_list');C=pl.col
torch.set_num_threads(2)


def run():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'report.json').exists(),'Completed experiment must remain frozen'
    refroot=Path('artifacts/evidence_session62_grounded_list_boost')
    reference=joblib.load(available(refroot/'list_boost_full_fold2.joblib'))
    cfg=dict(reference['config']);gc=reference['grounded_columns'];jc=['jev_'+k for k in QUESTIONS]
    b.CONFIG.update(cfg);b.features=features
    d=data().join(pl.read_parquet('artifacts/evidence_session195_grounded_selection/grounded.parquet'),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(pl.read_parquet('artifacts/evidence_session198_jev_full/oof.parquet').select('pair_id','hand_id',*jc),on=['pair_id','hand_id'],validate='1:1')
    (ROOT/'config.json').write_text(json.dumps({'method':__doc__,'config':cfg,'extra_columns':jc,'grounded_columns':gc},indent=2))
    outputs=[];audits=[];start=time.monotonic()
    expected=pl.read_parquet(available(refroot/'conditional_boost_oof.parquet'))
    for f in range(4):
        groups,compact,extra,P,M,T,V,D,fv,bid,pos=b.pack(d,f,gc+jc)
        X=np.nan_to_num(np.column_stack([compact,extra]),nan=0,posinf=1e6,neginf=-1e6)
        train=np.flatnonzero((fv!=f)&V.any(1).numpy());valid=np.flatnonzero(fv==f)
        rt=np.isin(bid,train);rv=np.isin(bid,valid);ti=torch.tensor(train)
        fam=np.array([g['behavior_family'][0] for g in groups])
        minimums={s:int(D.numpy()[train[fam[train]==s]].min()) for s in set(fam)}
        K=torch.tensor([minimums[s] for s in fam]);rowweight=1/D.numpy()[bid]
        assert not set(d.filter(C('fold')!=f)['table_id'])&set(d.filter(C('fold')==f)['table_id'])
        probe=torch.zeros_like(P,requires_grad=True);loss=old.objective(P,probe,M,T,V,D,ti,K);loss.backward()
        assert not torch.count_nonzero(probe.grad[valid])
        tt=T.clone();tt[valid]=1;dd=D.clone();dd[valid]=999
        assert float(loss.detach())==float(old.objective(P,probe.detach(),M,tt,V,dd,ti,K))
        A=torch.zeros_like(P);initial=float(old.objective(P,A,M,T,V,D,ti,K));history=[]
        model={'steps':[],'fold':f,'kind':'jev_full','config':cfg,'minimums':minimums,'grounded_columns':gc,
               'jev_columns':jc,'feature_count':X.shape[1]}
        for it in range(cfg['iterations']):
            A.requires_grad_(True);loss=old.objective(P,A,M,T,V,D,ti,K);loss.backward()
            grad=A.grad.detach().numpy()[bid,pos];a=A.detach().numpy()[bid,pos]
            prob=torch.softmax(torch.cat([torch.zeros_like(P[:,:,:1]),P+A.detach()],2),2).numpy()[bid,pos,1:]
            h=np.maximum(prob*(1-prob),.02)*rowweight[:,None]+2*cfg['ridge']/M.sum(1).numpy()[bid,None]
            target=np.clip(-grad/h,-5,5)
            tree=DecisionTreeRegressor(max_leaf_nodes=cfg['max_leaf_nodes'],min_samples_leaf=cfg['min_samples_leaf'],
                                       max_features=cfg['max_features'],random_state=cfg['seed']+f*1000+it)
            tree.fit(X[rt],target[rt],sample_weight=h[rt].mean(1));update=tree.predict(X).astype(np.float32)
            prior_loss=float(loss.detach());A=A.detach();step=cfg['learning_rate'];accepted=False
            with torch.no_grad():
                for _ in range(7):
                    candidate=torch.zeros_like(P);candidate[bid,pos]=torch.tensor(np.clip(a+step*update,-cfg['bound'],cfg['bound']))
                    new=float(old.objective(P,candidate,M,T,V,D,ti,K))
                    if new<=prior_loss+1e-6:accepted=True;break
                    step*=.5
            if not accepted:step=0.;candidate=A;new=prior_loss
            A=candidate;model['steps'].append((tree,step));history.append(new)
            if (it+1)%30==0:print('fold',f,'iteration',it+1,'seconds',round(time.monotonic()-start,1),flush=True)
        path=ROOT/f'fold{f}.joblib';joblib.dump(model,path,compress=3)
        np.testing.assert_allclose(b.infer(joblib.load(path),X[rv]),A.numpy()[bid[rv],pos[rv]],atol=1e-6,rtol=0)
        control_path=refroot/f'list_boost_full_fold{f}.joblib';cz=None
        if not control_path.stat().st_flags&0x40000000:
            control=joblib.load(control_path);cz=P.numpy().copy();cz[bid,pos]+=b.infer(control,X[:,:-len(jc)])
        replay=0.
        for i in valid:
            g=groups[i];n=len(g);z=(P+A)[i,:n].numpy()
            prob=torch.softmax(torch.tensor(np.column_stack([np.zeros(n,np.float32),z])),1).numpy()
            inc=conditioned(prob[:,1:],minimums[g['behavior_family'][0]])
            score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
            reference_score=g.select('pair_id','hand_id').join(expected.select('pair_id','hand_id','full'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['full'].to_numpy()
            if cz is not None:
                prob0=torch.softmax(torch.tensor(np.column_stack([np.zeros(n,np.float32),cz[i,:n]])),1).numpy()
                inc0=conditioned(prob0[:,1:],control['minimums'][g['behavior_family'][0]])
                replay=max(replay,float(abs(reference_score-(.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc0)).max()))
            outputs.append(g.select('pair_id','hand_id').with_columns(pl.Series('jev_list',score),pl.Series('reference_list',reference_score)))
        assert replay<1e-7,replay
        audits.append({'fold':f,'training_loss':history,'initial_loss':initial,'final_loss':history[-1],
                       'checkpoint_replay':True,'heldout_gradients_zero':True,'heldout_labels_training_invariant':True,
                       'control_checkpoint_available':cz is not None,'control_replay_error':replay if cz is not None else None})
        (ROOT/'audit.json').write_text(json.dumps(audits,indent=2));pl.concat(outputs).write_parquet(ROOT/'partial_oof.parquet')
    q=pl.concat(outputs);q.write_parquet(ROOT/'oof.parquet')
    d=d.join(q,on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id','equal','pressure59'),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id','conditional_family'),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score'),on='pair_id',validate='m:1')
    gate=d['risk_score'].to_numpy()>=.05;fallback=d['conditional_family'].to_numpy()
    pred={'r33':np.where(gate,d['equal'].to_numpy(),fallback),
          'reference_list':np.where(gate,d['reference_list'].to_numpy(),fallback),
          'jev_list':np.where(gate,d['jev_list'].to_numpy(),fallback),
          'jev_equal_pressure':np.where(gate,.5*(d['jev_list'].to_numpy()+d['pressure59'].to_numpy()),fallback)}
    reporting.ROOT=ROOT;report=reporting.metrics(d.drop('row'),pred)
    assert abs(report['r33']['MAP']-.7973334826762246)<1e-12
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
