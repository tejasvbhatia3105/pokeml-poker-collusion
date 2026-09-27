\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json, hashlib, time, shutil
from pathlib import Path
import numpy as np
import polars as pl
from scipy.special import expit, logit
from catboost import CatBoostClassifier, Pool
from session48_candidate_selection import design, CONFIG as OLD_CONFIG

ROOT=Path('artifacts/evidence_session189_pair_event_prototypes')
C=pl.col
FIELDS=['equity','partner_private_equity','partner_equity_minus_own','players_active',
        'street_no','pot_odds','call_stack','partner_alive','partner_last_aggressor',
        'response_log_amount_bb','response_amount_pot_fraction',
        'response_raise_increment_pot_fraction','response_call_coverage',
        'mw_own','mw_partner','mw_team','mw_call_edge','mw_information_gap',
        'mw_partner_fold_gain','mw_fold_value']
HEADS=['cat_primary','cat_secondary','hist_primary','hist_secondary']
CONFIG={'method':__doc__,'fields':FIELDS,'classes':[0,1,2,3],
        'bandwidth':'8th nearest OTHER eligible hand; mean squared standardized distance; floor .25',
        'standardization':'outer-training hands only; signed log1p, median/IQR with floor .1; clip +/-6',
        'iterations':400,'depth':4,'learning_rate':.03,'l2_leaf_reg':20,
        'seed':'4810+fold','input_root':'artifacts/evidence_session55_current_nested',
        'shortlist':12,'arms':['raw','prototype'],
        'transport':'explicit prior-shift check, no weight/family search',
        'limits':['Public validation pools reused heavily','No claimed leaderboard improvement']}

def data():
    d=pl.read_parquet('artifacts/policy/evidence_training.parquet')
    d=d.join(pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id','evidence_rank'),on=['pair_id','hand_id'],how='left',validate='1:1')
    d=d.sort('pair_id','time','hand_id').with_row_index('row')
    assert len(d)==45129 and d['evidence'].sum()==1817
    return d

def action_means(d):
    src=Path('artifacts/evidence_session163_generic_action_data')
    cfg=json.loads((src/'config.json').read_text())
    meta=pl.read_parquet(src/'actions.parquet')
    hand=pl.read_parquet('artifacts/evidence_session122_family_blind_hands/hands.parquet')
    x=np.load(src/'x.npy',mmap_mode='r')[:,[cfg['columns'].index(c) for c in FIELDS]]
    ids=hand.select('hand_index','pair_id','hand_id').join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='1:1').sort('hand_index')['row'].to_numpy()
    bag=ids[meta['hand_index'].to_numpy()]
    cls=meta['action_class'].to_numpy()
    result=np.zeros((len(d),4,len(FIELDS)),np.float64);count=np.zeros((len(d),4))
    for k in range(4):
        mask=cls==k;np.add.at(result[:,k],bag[mask],x[mask]);count[:,k]=np.bincount(bag[mask],minlength=len(d))
        result[:,k]/=np.maximum(1,count[:,k,None])
                                                                      
    for i in np.random.default_rng(189).choice(len(d),64,replace=False):
        for k in range(4):
            z=x[(bag==i)&(cls==k)]
            np.testing.assert_allclose(result[i,k],z.mean(0,dtype=np.float64) if len(z) else np.zeros(len(FIELDS)),rtol=1e-6,atol=1e-6)
            assert count[i,k]==len(z)
    return result.astype(np.float32),count

def prototypes(x,valid,p):
    \
    n=len(x);out=[]
    for k in range(4):
        distance=np.mean((x[:,None,k]-x[None,:,k])**2,axis=2,dtype=np.float64)
        allowed=np.broadcast_to(valid[:,k][None,:],(n,n)).copy();np.fill_diagonal(allowed,False)
        masked=np.where(allowed,distance,np.inf)
        nb=allowed.sum(1);j=np.minimum(7,np.maximum(nb-1,0))
        bandwidth=np.sort(masked,axis=1)[np.arange(n),j]
        bandwidth=np.where(np.isfinite(bandwidth),np.maximum(.25,bandwidth),1.)
        w=np.exp(-distance/bandwidth[:,None])*allowed
        total=w.sum(1);mean=w@p/np.maximum(total,1e-30)[:,None]
        global_mean=allowed@p/np.maximum(nb,1)[:,None]
        var=w@(p*p)/np.maximum(total,1e-30)[:,None]-mean*mean
        ratio=logit(mean.clip(1e-5,1-1e-5))-logit(global_mean.clip(1e-5,1-1e-5))
        eff=total**2/np.maximum((w*w).sum(1),1e-30)
        block=np.column_stack([mean,ratio,np.sqrt(np.maximum(var,0)),np.log1p(nb),np.log1p(eff)])
        block[(nb==0)|~valid[:,k]]=0
        out.append(block)
    return np.column_stack(out).astype(np.float32)

def score_report(d,pred):
    rows=[]
    for _,g in d.group_by('pair_id',maintain_order=True):
        ix=g['row'].to_numpy();y=g['evidence'].to_numpy();hand=g['hand_id'].to_numpy();den=min(5,int(y.sum()))
        row={'pair_id':g['pair_id'][0],'table_id':g['table_id'][0],'fold':int(g['fold'][0]),'family':g['behavior_family'][0]}
        for name,p in pred.items():
            order=np.lexsort((hand,-p[ix]))[:5];v=y[order]
            row[name]=float(np.sum(v*np.cumsum(v)/np.arange(1,len(v)+1))/den)
        rows.append(row)
    pairs=pl.DataFrame(rows);pairs.write_csv(ROOT/'pairs.csv')
    names=list(pred);pool=pairs.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id')
    boot=np.random.default_rng(18901).integers(0,len(pool),(5000,len(pool)))
    report={'results':{},'matched_prototype_vs_raw':{}}
    for name in names:
        delta=pool[name].to_numpy()-pool['r33'].to_numpy();bs=delta[boot].sum(1)/pool['n'].to_numpy()[boot].sum(1)
        report['results'][name]={'MAP':pairs[name].mean(),'gain_vs_r33':pairs[name].mean()-pairs['r33'].mean(),'pool_CI95':np.quantile(bs,[.025,.975]).tolist(),
            'folds':pairs.group_by('fold').agg(C(name).mean()).sort('fold')[name].to_list(),
            'families':dict(pairs.group_by('family').agg(C(name).mean()).iter_rows()),
            'improved':int((pairs[name]>pairs['r33']).sum()),'worse':int((pairs[name]<pairs['r33']).sum())}
    for a,b in [('prototype','raw'),('prototype_transport','raw_transport')]:
        delta=pool[a].to_numpy()-pool[b].to_numpy();bs=delta[boot].sum(1)/pool['n'].to_numpy()[boot].sum(1)
        report['matched_prototype_vs_raw'][a]={'gain':pairs[a].mean()-pairs[b].mean(),'CI95':np.quantile(bs,[.025,.975]).tolist()}
    assert abs(report['results']['r33']['MAP']-.7973334826762246)<1e-12
    return report

def main():
    ROOT.mkdir(exist_ok=True)
    assert shutil.disk_usage('.').free>10*2**30
    (ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));start=time.time();d=data()
    raw,count=action_means(d);trans=np.sign(raw)*np.log1p(abs(raw));valid=count>0;fv=d['fold'].to_numpy();y=d['evidence'].to_numpy()
    old=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['equal'].to_numpy()
    route=d.select('pair_id').join(pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score'),on='pair_id',validate='m:1',maintain_order='left')['risk_score'].to_numpy()>=.05
    r30=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['conditional_family'].to_numpy()
    pred={'r33':np.where(route,old,r30)}
    for name in ['compact_control','raw','prototype','raw_transport','prototype_transport']:pred[name]=np.zeros(len(d))
    groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];audit=[];proof=[]
    for f in range(4):
        compact,prior,selected,minimums=design(d,f,CONFIG['input_root'])
        fixture=d.select('pair_id','hand_id').join(pl.read_parquet(f'artifacts/evidence_session66_current_candidates/design_fold{f}.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
        np.testing.assert_array_equal(prior,fixture['prior'].to_numpy());np.testing.assert_array_equal(selected,fixture['selected'].to_numpy())
        tr=(fv!=f)&selected;va=fv==f;sv=va&selected;p0=logit(prior.clip(1e-7,1-1e-7))
        z=np.zeros_like(trans);norm=[]
        for k in range(4):
            fit=trans[(fv!=f)&valid[:,k],k];mu=np.median(fit,axis=0);lo,hi=np.quantile(fit,[.25,.75],axis=0);sd=np.maximum(.1,hi-lo)
            z[:,k]=np.clip((trans[:,k]-mu)/sd,-6,6);z[~valid[:,k],k]=0;norm.append({'mu':mu.tolist(),'sd':sd.tolist()})
        teachers=d.select('pair_id','hand_id').join(pl.read_parquet(f"{CONFIG['input_root']}/nested_outer{f}.parquet"),on=['pair_id','hand_id'],validate='1:1',maintain_order='left').select(HEADS).to_numpy()
        context=np.zeros((len(d),56),np.float32)
        for j,ix in enumerate(groups):
            context[ix]=prototypes(z[ix],valid[ix],teachers[ix])
            if j==0:
                rev=np.arange(len(ix))[::-1];check=prototypes(z[ix][rev],valid[ix][rev],teachers[ix][rev]);np.testing.assert_allclose(check[::-1],context[ix],atol=1e-6)
                mutated=teachers[ix].copy();mutated[0]=[.99,.01,.8,.2];check=prototypes(z[ix],valid[ix],mutated);np.testing.assert_array_equal(check[0],context[ix][0])
        x=np.column_stack([compact,trans.reshape(len(d),-1),np.log1p(count)]).astype(np.float32)
        control=CatBoostClassifier();control.load_model(f'artifacts/evidence_session66_current_candidates/compact_fold{f}.cbm')
        pc=prior[va].copy();pc[selected[va]]=expit(p0[sv]+control.predict(compact[sv],prediction_type='RawFormulaVal',thread_count=2));pred['compact_control'][va]=pc
        for kind in CONFIG['arms']:
            xx=x if kind=='raw' else np.column_stack([x,context]);m=CatBoostClassifier(iterations=400,depth=4,learning_rate=.03,l2_leaf_reg=20,random_seed=4810+f,thread_count=2,verbose=False,allow_writing_files=False)
            m.fit(Pool(xx[tr],y[tr],baseline=p0[tr]));m.save_model(str(ROOT/f'{kind}_fold{f}.cbm'))
            delta=m.predict(xx[sv],prediction_type='RawFormulaVal',thread_count=2)
            replay=CatBoostClassifier();replay.load_model(str(ROOT/f'{kind}_fold{f}.cbm'));np.testing.assert_array_equal(delta,replay.predict(xx[sv],prediction_type='RawFormulaVal',thread_count=2))
            pred[kind][va]=prior[va];pred[kind][sv]=expit(p0[sv]+delta)
            pred[kind+'_transport'][va]=old[va];pred[kind+'_transport'][sv]=expit(logit(old[sv].clip(1e-7,1-1e-7))+delta)
            audit.append({'fold':f,'arm':kind,'features':xx.shape[1],'training_rows':int(tr.sum()),'training_pools':d.filter(pl.Series(tr))['table_id'].n_unique(),'heldout_rows':int(sv.sum()),'training_heldout_overlap':int((tr&va).sum()),'checkpoint_replay_exact':True})
            print('FIT',f,kind,round(time.time()-start,1),flush=True)
        (ROOT/f'norm_fold{f}.json').write_text(json.dumps(norm))
        proof.append({'fold':f,'frozen66_prior_and_shortlist_exact':True,'row_reversal_invariant':True,'self_probability_excluded':True})
    saved=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session66_current_candidates/oof.parquet').select('pair_id','hand_id','compact'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['compact'].to_numpy()
    np.testing.assert_allclose(pred['compact_control'],saved,atol=1e-12,rtol=0)
    replay_error=float(abs(pred['compact_control']-saved).max())
    for name in pred:pred[name]=np.where(route,pred[name],r30)
    d.select('pair_id','hand_id').with_columns(*[pl.Series(n,p) for n,p in pred.items()]).write_parquet(ROOT/'oof.parquet')
    report=score_report(d,pred);report['seconds']=time.time()-start;report['method']=__doc__;report['compact66_replay_error']=replay_error
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'verification.json').write_text(json.dumps(proof,indent=2));(ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
