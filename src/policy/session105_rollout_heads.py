\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,sys,time
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from session55_current_targets import state
from session35_fold_likelihood import labels
from session29_shared_fold_witness import target
import session39_bet_call as calls
from session25_persistent_actor import actor_features
import session57_isolation_pressure as iso
import session104_allstreet_rollout as roll
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session105_rollout_heads');C=pl.col;KINDS=['checkcall','learned']

def inputs():
    states=state();out={}
    for fam,kind in [('directed_transfer','direct_primary'),('soft_play','soft_primary')]:
        v=states[fam];out[kind]={'d':v['d'],'a':v['a'],'x':v['x'],'family':fam}
    d,a,_=calls.data();hc=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event']
    out['direct_secondary']={'d':d,'a':a,'x':d.select(hc).to_numpy(),'ax':actor_features(d),'family':'directed_transfer'}
    full,d,a,ac=iso.data();hc=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];g=a['row'].to_numpy()
    x=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x']])
    out['isolation']={'d':d,'a':a,'x':x,'full':full,'family':'coordinated_isolation'}
    return out

def donor_weights(d):
    return d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy()

def baseline():
    ROOT.mkdir(exist_ok=True);base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');records=[]
    for kind,v in inputs().items():
        d,a,x=v['d'],v['a'],v['x'];g=a['row'].to_numpy();fv=d['fold'].to_numpy();nh=2 if kind=='isolation' else 1
        actor=None if kind=='isolation' else a['actor'].to_numpy();pred=np.zeros((len(d),2))
        for f in range(4):
            if kind=='direct_secondary':
                va=fv==f;m=CatBoostClassifier();m.load_model(f'artifacts/evidence_session25_persistent_actor/oriented/event2_directed_transfer_fold{f}.cbm')
                for r in range(2):pred[va,r]=m.predict_proba(np.column_stack([x[va],v['ax'][va,r]]),thread_count=2)[:,1]
                continue
            va=fv[g]==f
            for head in range(nh):
                if kind in ['direct_primary','soft_primary']:path=Path('artifacts/evidence_session50_matchup/current')/v['family']/f'event1_fold{f}.cbm'
                elif kind=='direct_secondary':path=calls.ROOT/f'secondary_fold{f}_em2.cbm'
                else:path=Path('artifacts/evidence_session59_pressure_equity')/f'event{head+1}_fold{f}_em2.cbm'
                m=CatBoostClassifier();m.load_model(str(path));p=m.predict_proba(x[va],thread_count=2)[:,1]
                if kind=='isolation':pred[fv==f,head]=iso.noisy_or(p,g[va],len(d))[fv==f]
                elif kind=='direct_secondary':pred[fv==f]=iso.noisy_or(p,2*g[va]+actor[va],2*len(d)).reshape(-1,2)[fv==f]
                else:pred[g[va],actor[va]]=p
        expected=d.select('pair_id','hand_id').join(base,on=['pair_id','hand_id'],maintain_order='left',validate='1:1')
        if kind=='isolation':one=pred;two=expected.select('bg_primary','bg_secondary').to_numpy()
        else:
            one=(pred*(donor_weights(d) if kind.startswith('direct') else 1)).sum(1)
            two=expected['bg_secondary' if kind=='direct_secondary' else 'bg_primary'].to_numpy()
        np.testing.assert_array_equal(one,two);records.append({'kind':kind,'actions':len(a),'baseline_error':0,'heads':nh})
    (ROOT/'baseline_replay.json').write_text(json.dumps(records,indent=2));print(records)

def prepare():
    ROOT.mkdir(exist_ok=True);q=pl.read_parquet(roll.ROOT/'queries.parquet');cols=[a+'_'+c for a in KINDS for c in roll.FIELDS];xs=np.zeros((4,len(q),len(cols)));seen=np.zeros((4,len(q)),int)
    config=json.loads((roll.ROOT/'config.json').read_text());assert config['provenance_sha256']==roll.provenance()
    for (table,),local in q.group_by('table_id'):
        native=int(local['fold'][0]);ix=local['query_id'].to_numpy()
        for f in range(4):
            if f==native:continue
            a,b=sorted([f,native]);path=roll.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet'
            audit=json.load(open(path.with_suffix('.json')));assert audit['config']==config
            z=local.select('query_id').join(pl.read_parquet(path),on='query_id',validate='1:1',maintain_order='left')
            value=z.select(cols).to_numpy();assert np.isfinite(value).all()
            xs[f,ix]=value;seen[f,ix]+=1;xs[native,ix]+=value/3;seen[native,ix]+=1
    for f in range(4):
        np.testing.assert_array_equal(seen[f],np.where(q['fold'].to_numpy()==f,3,1))
        np.savez_compressed(ROOT/f'features_fold{f}.npz',x=xs[f])
    q.write_parquet(ROOT/'queries.parquet');(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'fields':cols,'rollout_config':config,'arms':KINDS,'schedule':'unchanged original head schedules; no early stopping or tuning'},indent=2))

def extra(a,kind,f,arm):
    q=pl.read_parquet(ROOT/'queries.parquet').filter(C('kind')==kind)
    z=a.select('pair_id','hand_id',C('action_no').cast(pl.Int64)).join(q.select('pair_id','hand_id','action_no','query_id'),on=['pair_id','hand_id','action_no'],maintain_order='left',validate='1:1')
    assert z['query_id'].null_count()==0 and len(z)==len(a)
    xs=np.load(ROOT/f'features_fold{f}.npz')['x'];offset=KINDS.index(arm)*len(roll.FIELDS)
    return xs[z['query_id'].to_numpy(),offset:offset+len(roll.FIELDS)]

def training(v,kind,f,head):
    d,a=v['d'],v['a'];g=a['row'].to_numpy();fv=d['fold'].to_numpy();va=fv[g]==f
    if kind=='direct_primary':y,tr,_=labels(d,a,f);return y,tr,va,None
    if kind=='soft_primary':yy,e,_=target(d,f);return yy[g],e[g],va,None
    assert kind=='isolation'
    source=iso.targets(v['full'],f,'coordinated_isolation');fm=v['full']['behavior_family'].to_numpy()=='coordinated_isolation'
    y=source[head][fm];tr=source[head+2][fm][g]
    return y[g],tr,va,g[tr]

def secondary_features(v,f,arm):
    d,a=v['d'],v['a'];z=extra(a,'direct_secondary',f,arm);g=2*a['row'].to_numpy()+a['actor'].to_numpy()
    count=np.bincount(g,minlength=2*len(d));total=np.zeros((2*len(d),z.shape[1]));high=np.full_like(total,-np.inf)
    np.add.at(total,g,z);np.maximum.at(high,g,z);high[count==0]=0
    mean=total/np.maximum(count,1)[:,None]
    return np.column_stack([mean,high]).reshape(len(d),2,-1)

def train():
    base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');parts={a:[] for a in KINDS};audit=[];start=time.time()
    for kind,v in inputs().items():
        d,a=v['d'],v['a'];g=a['row'].to_numpy();fv=d['fold'].to_numpy();nh=2 if kind=='isolation' else 1
        actor=None if kind=='isolation' else a['actor'].to_numpy()
        if kind=='direct_secondary':
            for arm in KINDS:
                dest=ROOT/arm;dest.mkdir(exist_ok=True);pred=np.zeros((len(d),2))
                for f in range(4):
                    ex=secondary_features(v,f,arm);y,tr,donor=calls.targets(d,f);va=fv==f;ix=np.flatnonzero(tr)
                    assert not(tr&va).any() and (donor[tr]>=0).all()
                    x=np.column_stack([v['x'][tr],v['ax'][ix,donor[tr]],ex[ix,donor[tr]]])
                    m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6312+11*f,thread_count=2,allow_writing_files=False,verbose=False)
                    m.fit(x,y[tr]);m.save_model(str(dest/f'{kind}_fold{f}_head0_em0.cbm'))
                    for r in range(2):pred[va,r]=m.predict_proba(np.column_stack([v['x'][va],v['ax'][va,r],ex[va,r]]),thread_count=2)[:,1]
                    audit.append({'kind':kind,'arm':arm,'fold':f,'head':0,'training_hands':int(tr.sum()),'validation_hands':int(va.sum()),'validation_overlap':0,'EM_steps':1,'elapsed':time.time()-start})
                    (ROOT/'fit_audit.json').write_text(json.dumps(audit,indent=2));print('HEAD_DONE',kind,arm,f,0,time.time()-start,flush=True)
                parts[arm].append(d.select('pair_id','hand_id').with_columns(pl.Series('new_secondary',(pred*donor_weights(d)).sum(1))))
            continue
        for arm in KINDS:
            dest=ROOT/arm;dest.mkdir(exist_ok=True);pred=np.zeros((len(d),2))
            for f in range(4):
                x=np.column_stack([v['x'],extra(a,kind,f,arm)])
                for head in range(nh):
                    y,tr,va,bag=training(v,kind,f,head);assert not(tr&va).any()
                    if bag is None:response=y[tr];steps=1
                    else:count=np.bincount(bag,minlength=len(d));response=y[tr]/count[bag];steps=3
                    for em in range(steps):
                        seed=6311+11*f+(1 if kind=='direct_secondary' else head)
                        m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,
                            loss_function='CrossEntropy' if bag is not None else 'Logloss',random_seed=seed,
                            thread_count=2,allow_writing_files=False,verbose=False)
                        m.fit(x[tr],response);m.save_model(str(dest/f'{kind}_fold{f}_head{head}_em{em}.cbm'))
                        if bag is not None:
                            p=m.predict_proba(x[tr],thread_count=2)[:,1];hp=iso.noisy_or(p,bag,len(d));response=np.where(y[tr],p/np.maximum(hp[bag],1e-8),0).clip(0,1)
                    p=m.predict_proba(x[va],thread_count=2)[:,1]
                    if kind=='isolation':pred[fv==f,head]=iso.noisy_or(p,g[va],len(d))[fv==f]
                    elif kind=='direct_secondary':pred[fv==f]=iso.noisy_or(p,2*g[va]+actor[va],2*len(d)).reshape(-1,2)[fv==f]
                    else:pred[g[va],actor[va]]=p
                    audit.append({'kind':kind,'arm':arm,'fold':f,'head':head,'training_actions':int(tr.sum()),'validation_actions':int(va.sum()),'validation_overlap':0,'EM_steps':steps,'elapsed':time.time()-start})
                    (ROOT/'fit_audit.json').write_text(json.dumps(audit,indent=2));print('HEAD_DONE',kind,arm,f,head,time.time()-start,flush=True)
            if kind=='isolation':
                parts[arm].append(d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',pred[:,0]),pl.Series('new_secondary',pred[:,1])))
            else:
                p=(pred*(donor_weights(d) if kind.startswith('direct') else 1)).sum(1)
                col='new_secondary' if kind=='direct_secondary' else 'new_primary'
                parts[arm].append(d.select('pair_id','hand_id').with_columns(pl.Series(col,p)))
    for arm in KINDS:
        primary=pl.concat([z.select('pair_id','hand_id','new_primary') for z in parts[arm] if 'new_primary' in z.columns])
        secondary=pl.concat([z.select('pair_id','hand_id','new_secondary') for z in parts[arm] if 'new_secondary' in z.columns])
        out=base.join(primary,on=['pair_id','hand_id'],how='left',validate='1:1').join(secondary,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary'),pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_primary','new_secondary')
        assert len(out)==len(base);out.write_parquet(ROOT/arm/'event_oof.parquet');assemble(ROOT/arm)

if __name__=='__main__':{'baseline':baseline,'prepare':prepare,'train':train}[sys.argv[1]]()
