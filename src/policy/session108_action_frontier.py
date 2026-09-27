\
\
\
\
\
\
\
\
import gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import polars as pl
import session104_allstreet_rollout as s

ROOT=Path('artifacts/evidence_session108_action_frontier');C=pl.col
FIELDS=['own_regret','partner_gain_own','team_gain_own','team_regret',
        'own_gain_team','partner_gain_team','partner_regret','own_gain_partner',
        'team_gain_partner','actual_own_rate','actual_team_rate','actual_partner_rate']
CONFIG={'method':__doc__,'replicates':64,'root_batch':8,
        'menu':'observed; fold/check; call; minimum legal raise; raise by pot after call; all-in; duplicate actions collapsed',
        'selection':'choose action on paths 0:32, evaluate 32:64; swap; average both evaluation halves',
        'reference_exclusion':'same three native/outer fold exclusions as104',
        'scaling':'all values divided by root pot; own_regret positive when alternative gains more own value',
        'arms':['checkcall','learned'],'random':'same per-replicate grid and future-board scheme as104',
        'tie_rule':'first menu action within 1e-10 normalized-pot units of the largest selection-half mean'}

def provenance():
    out=s.provenance();p=Path(__file__);out[str(p.relative_to(Path.cwd())) if p.is_absolute() else str(p)]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    return out

def menu(root):
    st=root['state'];j=root['own'];call=st.to_call(j)
    result=[(int(root['forced_class']),float(root['forced_amount']))]
    result.append((0 if call>0 else 1,0.))
    if call>0:result.append((2,float(call)))
    if st.remaining[j]>call and st.raise_right[j] and np.any(st.alive&(st.remaining>0)&(np.arange(6)!=j)):
        low=min(st.remaining[j],call+st.minimum_raise)
                                                                                  
        pot=min(st.remaining[j],max(low,float(np.floor(st.contribution.sum()+2*call))))
        result.extend((3,float(x)) for x in [low,pot,st.remaining[j]])
    result=list(dict.fromkeys(result))
    for k,amount in result:assert not st.clone().apply(j,k,amount)
    return result

def simulate(roots,action,sizer,meta,reps=64):
    \
    u=np.random.default_rng(10401).random((reps,s.CONFIG['max_actions'],2));states=[];ri=[];sc=[];rep=[]
    for i,r in enumerate(roots):
        for arm in range(2):
            for k in range(reps):states.append(r['state'].clone());ri.append(i);sc.append(arm);rep.append(k)
    ri=np.array(ri);sc=np.array(sc);rep=np.array(rep);centers=np.array(meta['centers']);total=0;maxstep=0
    for step in range(s.CONFIG['max_actions']):
        pairs=[(i,st.next_decision()) for i,st in enumerate(states)];pairs=[(i,j) for i,j in pairs if j is not None]
        if not pairs:break
        ids=np.array([i for i,j in pairs]);actors=np.array([j for i,j in pairs]);maxstep=step+1
        kinds=np.array([2 if states[i].to_call(j)>0 else 1 for i,j in pairs]);amounts=np.array([states[i].to_call(j) for i,j in pairs])
        nn=np.flatnonzero(sc[ids]==0) if step else np.array([],int)
        if len(nn):
            xx=np.stack([s.inputs.vector(states[ids[n]],int(actors[n]),roots[ri[ids[n]]]['templates'][rep[ids[n]],states[ids[n]].street,actors[n]]) for n in nn])
            pp=action.predict_proba(xx,thread_count=2);legal=np.ones_like(pp,bool)
            for ii,n in enumerate(nn):
                st=states[ids[n]];j=int(actors[n]);call=st.to_call(j)
                legal[ii]=[call>0,call==0,call>0,st.remaining[j]>call and st.raise_right[j] and np.any(st.alive&(st.remaining>0)&(np.arange(6)!=j))]
            pp=np.maximum(pp,1e-12)*legal;pp/=pp.sum(1)[:,None]
            kk=(u[rep[ids[nn]],step,0,None]>np.cumsum(pp,axis=1)).sum(1).clip(0,3)
            kinds[nn]=kk;amounts[nn]=np.where(kk==2,amounts[nn],0);raisepos=np.flatnonzero(kk==3)
            if len(raisepos):
                sp=sizer.predict_proba(xx[raisepos],thread_count=2);raises=nn[raisepos]
                chosen=(u[rep[ids[raises]],step,1,None]>np.cumsum(sp,axis=1)).sum(1).clip(0,sp.shape[1]-1)
                for n,k in zip(raises,chosen):
                    st=states[ids[n]];j=int(actors[n]);want=st.remaining[j] if k==len(centers)-1 else np.floor(max(0.,np.exp(centers[k])-.01)*max(st.contribution.sum(),1.))
                    amounts[n]=min(st.remaining[j],max(want,st.to_call(j)+st.minimum_raise))
        if step==0:
            for n,i in enumerate(ids):kinds[n]=roots[ri[i]]['forced_class'];amounts[n]=roots[ri[i]]['forced_amount']
        for i,j,k,amount in zip(ids,actors,kinds,amounts):assert not states[i].apply(int(j),int(k),float(amount))
        total+=len(ids)
    else:raise RuntimeError('128-action limit; no truncated values exported')
    out=np.zeros((len(roots),2,reps,6));error=0.
    for i,st in enumerate(states):
        assert st.next_decision() is None
        gross=s.payout(st.contribution,st.alive,roots[ri[i]]['ranks'][rep[i]]);assert gross is not None
        net=gross-st.contribution;error=max(error,float(abs(net.sum())));out[ri[i],sc[i],rep[i]]=net
    assert error<1e-8
    return out,dict(trajectories=len(states),actions=total,max_steps=maxstep,chip_conservation_error=error)

def evaluate(roots,action,size,meta,reps=64):
    expanded=[];offset=[0];menus=[]
    for r in roots:
        choices=menu(r);menus.append(choices)
        expanded.extend(dict(r,forced_class=k,forced_amount=amount) for k,amount in choices);offset.append(len(expanded))
    v,info=simulate(expanded,action,size,meta,reps);rows=[]
    for i,r in enumerate(roots):
        value=v[offset[i]:offset[i+1]]/max(1.,r['pot']);own=r['own'];half=reps//2
        assert half*2==reps
        for member in r['members']:
            other=member['other'];row={'query_id':member['query_id']}
            for arm,j in [('learned',0),('checkcall',1)]:
                vv=value[:,j];result=np.zeros(12)
                for select,test in [(slice(0,half),slice(half,reps)),(slice(half,reps),slice(0,half))]:
                    means=vv[:,select].mean(1);scores=[means[:,own],means[:,own]+means[:,other],means[:,other]]
                    chosen=[int(np.flatnonzero(z>=z.max()-1e-10)[0]) for z in scores]
                    delta=[(vv[0,test]-vv[k,test]).mean(0) for k in chosen]
                    a,b,c=delta
                    result+=np.array([-a[own],a[other],a[own]+a[other],-b[own]-b[other],b[own],b[other],-c[other],c[own],c[own]+c[other],*[float(k==0) for k in chosen]])/2
                row.update({arm+'_'+k:float(x) for k,x in zip(FIELDS,result)})
            rows.append(row)
    return pl.DataFrame(rows),dict(menu_cases=len(expanded),**info),v,offset

def pilot():
    ROOT.mkdir(exist_ok=True);q=pl.read_parquet(s.ROOT/'queries.parquet');table=q.filter(C('fold')==0)['table_id'][0]
    roots=s.roots_for_table(table,q.filter(C('table_id')==table),64);roots=sum(([r for r in roots if r['state'].street==street][:2] for street in range(4)),[])
    action,size,meta=s.models(0,1);start=time.time();z,info,v,offset=evaluate(roots,action,size,meta,64)
    old,_=s.simulate(roots,action,size,meta,64);np.testing.assert_array_equal(v[np.array(offset[:-1])],old[:,[0,2]])
    rev,_,_,_=evaluate(roots[::-1],action,size,meta,64);np.testing.assert_array_equal(z.sort('query_id').to_numpy(),rev.sort('query_id').to_numpy())
                                                                                        
    short,_=simulate(roots,action,size,meta,16);np.testing.assert_array_equal(short,old[:,[0,2],:16])
    z.write_parquet(ROOT/'pilot_values.parquet');report=dict(roots=len(roots),streets=[r['state'].street for r in roots],
        forced_arm_replay_error=0,query_reversal_error=0,simulator_prefix_error=0,seconds=time.time()-start,**info)
    (ROOT/'pilot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def initialize():
    ROOT.mkdir(exist_ok=True);(ROOT/'tables').mkdir(exist_ok=True);config=dict(CONFIG,provenance_sha256=provenance())
    cfg=ROOT/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:cfg.write_text(json.dumps(config,indent=2))
    q=pl.read_parquet(s.ROOT/'queries.parquet');q.write_parquet(ROOT/'queries.parquet')
    return q,config

def build():
    q,config=initialize();audits=[];start=time.time()
    for (table,),local in q.group_by('table_id',maintain_order=True):
        native=int(local['fold'][0]);pending=[]
        for f in range(4):
            if f==native:continue
            a,b=sorted([f,native]);path=ROOT/'tables'/f'{table}_exclude{a}{b}.parquet'
            if path.exists() and path.with_suffix('.json').exists():
                old=json.loads(path.with_suffix('.json').read_text());assert old['config']==config
                np.testing.assert_array_equal(pl.read_parquet(path).sort('query_id')['query_id'],local.sort('query_id')['query_id']);audits.append(old)
            else:pending.append((f,path))
        if not pending:continue
        roots=s.roots_for_table(table,local,CONFIG['replicates'])
        for f,path in pending:
            action,size,meta=s.models(native,f);assert table not in meta['training_tables'];parts=[];infos=[]
            for pos in range(0,len(roots),CONFIG['root_batch']):
                z,info,_,_=evaluate(roots[pos:pos+CONFIG['root_batch']],action,size,meta,CONFIG['replicates']);parts.append(z);infos.append(info)
            out=pl.concat(parts).sort('query_id');np.testing.assert_array_equal(out['query_id'],local.sort('query_id')['query_id']);assert np.isfinite(out.drop('query_id').to_numpy()).all();out.write_parquet(path)
            audit=dict(table_id=table,native_fold=native,other_excluded_fold=f,query_rows=len(local),roots=len(roots),config=config,
                actions=sum(x['actions'] for x in infos),trajectories=sum(x['trajectories'] for x in infos),menu_cases=sum(x['menu_cases'] for x in infos),
                chip_conservation_error=max(x['chip_conservation_error'] for x in infos),elapsed_seconds=time.time()-start)
            path.with_suffix('.json').write_text(json.dumps(audit,indent=2));audits.append(audit)
            print('FRONTIER_TABLE',table,'exclude',native,f,'roots',len(roots),'seconds',time.time()-start,flush=True)
        del roots;gc.collect()
    (ROOT/'build_audit.json').write_text(json.dumps(audits,indent=2))

if __name__=='__main__':{'pilot':pilot,'build':build,'initialize':initialize}[sys.argv[1]]()
