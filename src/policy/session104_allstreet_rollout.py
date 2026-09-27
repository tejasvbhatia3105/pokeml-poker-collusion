\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from cards import CARD,rank
import session101_betting_engine as engine
import session102_policy_inputs as inputs
from session33_terminal_replay import payout
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session104_allstreet_rollout');C=pl.col
FIELDS=['own_loss','partner_gain','team_gain','own_stderr','partner_stderr','team_stderr']
CONFIG={'method':__doc__,'replicates':32,'max_actions':128,'root_batch':32,
        'boards':'visible prefix and all12 dealt cards only; future draws seed10201, no observed future cards',
        'policies':'84 action model plus103 size distribution excluding native and one other fold',
        'outer_training':'for outer f/native h use excluded(f,h); validation f averages its3 excluded references',
        'control':'ordinary focal action, then check/call; compare same forced focal action',
        'styles':'original current-hand-excluded player/phase/street/time-bin counts reconstructed102',
        'random':'fixed seed10401 per replicate/step; no ID/order-dependent seeds',
        'scaling':'values and standard errors divided by root pot; positive own_loss is lost own value'}

def provenance():
    files=[Path('src/policy')/p for p in ['session101_betting_engine.py','session102_policy_inputs.py',
        'session103_nested_sizes.py','session104_allstreet_rollout.py','session34_river_engine.py','session33_terminal_replay.py','cards.py']]
    files += [Path('artifacts/policy')/p for p in ['feature_columns.json','preflop_equity.npy','cards.dylib']]
    for f in range(4):
        for h in range(f+1,4):
            files.extend([Path(f'artifacts/evidence_session84_nested_joint_policy/action_exclude{f}{h}.cbm'),
                          Path(f'artifacts/evidence_session103_nested_sizes/size_exclude{f}{h}.cbm'),
                          Path(f'artifacts/evidence_session103_nested_sizes/metadata_exclude{f}{h}.json')])
    return {str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files}

def queries():
    d=hand_data().select('pair_id','hand_id','table_id','fold','behavior_family')
    z=pl.read_parquet('artifacts/evidence_session29_shared_folds/fold_actions.parquet').select('pair_id','hand_id','action_no','street_no')
    z=z.join(d,on=['pair_id','hand_id'],validate='m:1').with_columns(pl.when(C('behavior_family')=='directed_transfer').then(pl.lit('direct_primary')).otherwise(pl.lit('soft_primary')).alias('kind'))
    parts=[z]
    for folder,filename,kind in [('evidence_session27_donor_calls','call_actions.parquet','direct_secondary'),('evidence_session57_isolation_pressure','pressure_actions.parquet','isolation')]:
        parts.append(pl.read_parquet(Path('artifacts')/folder/filename).select('pair_id','hand_id','action_no','street_no').join(d,on=['pair_id','hand_id'],validate='m:1').with_columns(pl.lit(kind).alias('kind')))
    q=pl.concat([p.select('pair_id','hand_id','table_id','fold','kind',C('action_no').cast(pl.Int64),C('street_no').cast(pl.Int64)) for p in parts])
    q=q.join(pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),on='pair_id',validate='m:1').sort('table_id','hand_id','action_no','kind','pair_id').with_row_index('query_id')
    assert q.select('kind','pair_id','hand_id','action_no').n_unique()==len(q)
    return q

def roots_for_table(table,q,reps=32):
    raw,meta,seats=engine.table_data(table,q.select('hand_id').unique())
    p=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet');styles=inputs.Styles(p)
    lookup={(hid,int(n)):g for (hid,n),g in q.group_by('hand_id','action_no')};roots=[];cache={}
    for (hid,),g in raw.group_by('hand_id',maintain_order=True):
        ss=seats[hid];people=ss['player_id'].to_list();st,index=engine.initialize(meta[hid],ss)
        holes=np.array([[CARD[a],CARD[b]] for a,b in ss.select('hole_card_1','hole_card_2').iter_rows()],np.int8)
        for row in g.to_dicts():
            j=index[row['player_id']];assert st.next_decision()==j
            matched=lookup.get((hid,row['action_no']))
            if matched is not None:
                key=hid,st.street
                if key not in cache:
                    nb=0 if st.street==0 else st.street+2
                    visible=[CARD[c] for c in meta[hid]['board_cards'].split()[:nb]]
                    boards,templates=inputs.future_templates(hid,people,holes,visible,st.street,styles,reps)
                    rr=np.array([[rank([*h,*b]) for h in holes] for b in boards])
                    cache[key]=boards,templates,rr
                board,template,rr=cache[key];members=[]
                for r in matched.to_dicts():
                    assert row['player_id'] in [r['player_1'],r['player_2']]
                    other=r['player_2'] if row['player_id']==r['player_1'] else r['player_1']
                    members.append({'query_id':r['query_id'],'other':index[other]})
                roots.append({'hand_id':hid,'action_no':row['action_no'],'state':st.clone(),'own':j,
                    'forced_class':engine.action_class(row),'forced_amount':row['amount'],
                    'templates':template,'ranks':rr,'boards':board,'pot':float(st.contribution.sum()),'members':members})
            assert not st.apply(j,engine.action_class(row),row['amount'])
    assert sum(len(r['members']) for r in roots)==len(q)
    return roots

def models(f,h):
    f,h=sorted([f,h]);ap=Path(f'artifacts/evidence_session84_nested_joint_policy/action_exclude{f}{h}.cbm');sp=Path(f'artifacts/evidence_session103_nested_sizes/size_exclude{f}{h}.cbm')
    action=CatBoostClassifier();action.load_model(str(ap));size=CatBoostClassifier();size.load_model(str(sp))
    meta=json.load(open(sp.with_name(f'metadata_exclude{f}{h}.json')))
    assert meta['excluded_folds']==[f,h]
    assert hashlib.file_digest(sp.open('rb'),'sha256').hexdigest()==meta['model_sha256']
    return action,size,meta

def simulate(roots,action,sizer,meta,reps=32):
    if not roots:return np.zeros((0,4,reps,6)),{}
    u=np.random.default_rng(10401).random((reps,CONFIG['max_actions'],2));states=[];ri=[];sc=[];rep=[]
    for i,r in enumerate(roots):
        for arm in range(4):
            for k in range(reps):states.append(r['state'].clone());ri.append(i);sc.append(arm);rep.append(k)
    ri=np.array(ri);sc=np.array(sc);rep=np.array(rep);centers=np.array(meta['centers']);total=0;street_counts=np.zeros(4,int);same=[];maxstep=0
    for step in range(CONFIG['max_actions']):
        pairs=[(i,st.next_decision()) for i,st in enumerate(states)];pairs=[(i,j) for i,j in pairs if j is not None]
        if not pairs:break
        ids=np.array([i for i,j in pairs]);actors=np.array([j for i,j in pairs]);maxstep=step+1
        xx=np.stack([inputs.vector(states[i],int(j),roots[ri[i]]['templates'][rep[i],states[i].street,j]) for i,j in pairs])
        kinds=np.where(xx[:,inputs.INDEX['call_bb']]>0,2,1);amounts=np.array([states[i].to_call(int(j)) for i,j in pairs])
        forced=(step==0)&np.isin(sc[ids],[0,2]);normal=((sc[ids]<2)|((step==0)&(sc[ids]==3)))&~forced;nn=np.flatnonzero(normal)
        if len(nn):
            pp=action.predict_proba(xx[nn],thread_count=2);legal=np.ones_like(pp,bool)
            for ii,n in enumerate(nn):
                i=ids[n];j=int(actors[n]);st=states[i];call=st.to_call(j)
                legal[ii]=[call>0,call==0,call>0,st.remaining[j]>call and st.raise_right[j] and np.any(st.alive&(st.remaining>0)&(np.arange(6)!=j))]
            pp=np.maximum(pp,1e-12)*legal;assert (pp.sum(1)>0).all();pp/=pp.sum(1)[:,None]
            kk=(u[rep[ids[nn]],step,0,None]>np.cumsum(pp,axis=1)).sum(1).clip(0,3)
            kinds[nn]=kk;amounts[nn]=np.where(kk==2,amounts[nn],0)
            raises=nn[kk==3]
            if len(raises):
                sp=sizer.predict_proba(xx[raises],thread_count=2)
                chosen=(u[rep[ids[raises]],step,1,None]>np.cumsum(sp,axis=1)).sum(1).clip(0,sp.shape[1]-1)
                for n,k in zip(raises,chosen):
                    i=ids[n];j=int(actors[n]);st=states[i]
                    want=st.remaining[j] if k==len(centers)-1 else np.floor(max(0.,np.exp(centers[k])-.01)*max(st.contribution.sum(),1.))
                    amounts[n]=min(st.remaining[j],max(want,st.to_call(j)+st.minimum_raise))
            if step==0:
                same=[int(ids[n]) for n in nn if kinds[n]==roots[ri[ids[n]]]['forced_class'] and amounts[n]==roots[ri[ids[n]]]['forced_amount']]
        for n in np.flatnonzero(forced):kinds[n]=roots[ri[ids[n]]]['forced_class'];amounts[n]=roots[ri[ids[n]]]['forced_amount']
        for i,j,k,amount in zip(ids,actors,kinds,amounts):
            street_counts[states[i].street]+=1
            failures=states[i].apply(int(j),int(k),float(amount));assert not failures,(failures,int(k),float(amount))
        total+=len(ids)
    else:raise RuntimeError('128-action rollout limit; no truncated values exported')
    result=np.zeros((len(roots),4,reps,6));error=0.
    for i,st in enumerate(states):
        assert st.next_decision() is None
        gross=payout(st.contribution,st.alive,roots[ri[i]]['ranks'][rep[i]])
        assert gross is not None
        net=gross-st.contribution;error=max(error,float(abs(net.sum())));result[ri[i],sc[i],rep[i]]=net
    for i in same:np.testing.assert_array_equal(result[ri[i],sc[i],rep[i]],result[ri[i],sc[i]-1,rep[i]])
    assert error<1e-8
    return result,{'trajectories':len(states),'actions':total,'street_actions':street_counts.tolist(),'max_steps':maxstep,'chip_conservation_error':error,'same_focal_action_exact_checks':len(same)}

def values(roots,result):
    rows=[]
    for i,r in enumerate(roots):
        for member in r['members']:
            own,other=r['own'],member['other'];row={'query_id':member['query_id']};den=max(1.,r['pot'])
            for arm,a,b in [('learned',0,1),('checkcall',2,3)]:
                delta=(result[i,a]-result[i,b])/den;team=delta[:,own]+delta[:,other]
                v=[-delta[:,own].mean(),delta[:,other].mean(),team.mean(),delta[:,own].std(ddof=1)/np.sqrt(len(delta)),delta[:,other].std(ddof=1)/np.sqrt(len(delta)),team.std(ddof=1)/np.sqrt(len(delta))]
                row.update({arm+'_'+k:float(x) for k,x in zip(FIELDS,v)})
            rows.append(row)
    return pl.DataFrame(rows)

def pilot():
    CONFIG['provenance_sha256']=provenance()
    ROOT.mkdir(exist_ok=True);q=queries();q.write_parquet(ROOT/'queries.parquet');(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    table=q.filter(C('fold')==0)['table_id'][0];local=q.filter(C('table_id')==table)
    start=time.time();roots=roots_for_table(table,local,reps=32);prep=time.time()-start
    selected=[]
    for street in range(4):selected.extend([r for r in roots if r['state'].street==street][:2])
    assert selected;action,size,meta=models(0,1);start=time.time();result,info=simulate(selected,action,size,meta,16)
    rev,_=simulate(selected[::-1],action,size,meta,16);np.testing.assert_array_equal(result,rev[::-1])
    short,_=simulate(selected,action,size,meta,8);np.testing.assert_array_equal(result[:,:,:8],short)
    info.update(table=table,query_counts=dict(q.group_by('kind').len().iter_rows()),total_query_rows=len(q),
                table_roots=len(roots),pilot_roots=len(selected),root_streets=[r['state'].street for r in selected],
                prepare_seconds=prep,three_simulations_seconds=time.time()-start,query_permutation_error=0,replicate_prefix_error=0)
    values(selected,result).write_parquet(ROOT/'pilot_values.parquet');(ROOT/'pilot.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

def build():
    CONFIG['provenance_sha256']=provenance()
    ROOT.mkdir(exist_ok=True);(ROOT/'tables').mkdir(exist_ok=True);q=queries();q.write_parquet(ROOT/'queries.parquet')
    cfg=ROOT/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==CONFIG
    else:cfg.write_text(json.dumps(CONFIG,indent=2))
    start=time.time();audits=[]
    for (table,),local in q.group_by('table_id',maintain_order=True):
        native=int(local['fold'][0]);pending=[]
        for f in range(4):
            if f==native:continue
            a,b=sorted([f,native]);path=ROOT/'tables'/f'{table}_exclude{a}{b}.parquet'
            if path.exists() and path.with_suffix('.json').exists():
                old=json.load(open(path.with_suffix('.json')));assert old['config']==CONFIG
                saved=pl.read_parquet(path);np.testing.assert_array_equal(saved.sort('query_id')['query_id'],local.sort('query_id')['query_id']);audits.append(old)
            else:pending.append((f,path))
        if not pending:continue
        roots=roots_for_table(table,local,CONFIG['replicates'])
        for f,path in pending:
            action,size,meta=models(f,native);assert table not in meta['training_tables'];parts=[];infos=[]
            for offset in range(0,len(roots),CONFIG['root_batch']):
                batch=roots[offset:offset+CONFIG['root_batch']];result,info=simulate(batch,action,size,meta,CONFIG['replicates']);parts.append(values(batch,result));infos.append(info)
            out=pl.concat(parts).sort('query_id');np.testing.assert_array_equal(out['query_id'],local.sort('query_id')['query_id'])
            assert np.isfinite(out.drop('query_id').to_numpy()).all();out.write_parquet(path)
            audit={'table_id':table,'native_fold':native,'other_excluded_fold':f,'query_rows':len(local),'roots':len(roots),'config':CONFIG,
                   'actions':sum(v['actions'] for v in infos),'trajectories':sum(v['trajectories'] for v in infos),
                   'chip_conservation_error':max(v['chip_conservation_error'] for v in infos),'elapsed_seconds':time.time()-start}
            path.with_suffix('.json').write_text(json.dumps(audit,indent=2));audits.append(audit)
            print('ROLLOUT_TABLE',table,'exclude',f,native,'roots',len(roots),'seconds',time.time()-start,flush=True)
            del action,size,result;gc.collect()
        del roots;gc.collect()
    (ROOT/'build_audit.json').write_text(json.dumps(audits,indent=2))

if __name__=='__main__':{'pilot':pilot,'build':build}[sys.argv[1]]()
