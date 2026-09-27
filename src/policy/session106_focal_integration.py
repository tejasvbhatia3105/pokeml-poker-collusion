\
\
\
\
\
\
\
import json,time
from pathlib import Path
import numpy as np
import polars as pl
import session104_allstreet_rollout as s
ROOT=Path('artifacts/evidence_session106_focal_integration');C=pl.col

def choices(root,action,sizer,meta):
    st=root['state'];j=root['own'];call=st.to_call(j)
    x=s.inputs.vector(st,j,root['templates'][0,st.street,j])[None]
    p=action.predict_proba(x,thread_count=2)[0]
    legal=np.array([call>0,call==0,call>0,st.remaining[j]>call and st.raise_right[j] and np.any(st.alive&(st.remaining>0)&(np.arange(6)!=j))])
    p=np.maximum(p,1e-12)*legal;p/=p.sum();out={}
    for k in range(3):
        if legal[k]:out[k,float(call if k==2 else 0.)]=p[k]
    if legal[3]:
        sp=sizer.predict_proba(x,thread_count=2)[0];centers=np.array(meta['centers'])
        for k,w in enumerate(sp):
            want=st.remaining[j] if k==len(centers)-1 else np.floor(max(0.,np.exp(centers[k])-.01)*max(st.contribution.sum(),1.))
            amount=float(min(st.remaining[j],max(want,call+st.minimum_raise)))
            out[3,amount]=out.get((3,amount),0.)+p[3]*w
    assert abs(sum(out.values())-1)<1e-12
    for (k,amount) in out:
        assert not st.clone().apply(j,k,amount)
    return out

def integrate(roots,action,size,meta,reps=128):
    expanded=[];mapping=[];weights=[]
    for i,root in enumerate(roots):
        for (k,amount),weight in choices(root,action,size,meta).items():
            expanded.append(dict(root,forced_class=k,forced_amount=amount));mapping.append(i);weights.append(weight)
    actual,info=s.simulate(roots,action,size,meta,reps)
    cases,more=s.simulate(expanded,action,size,meta,reps)
    normal=np.zeros((len(roots),2,reps,6))
    for r,i,w in zip(cases,mapping,weights):
        np.testing.assert_array_equal(r[1],actual[i,1]);np.testing.assert_array_equal(r[3],actual[i,3])
        normal[i,0]+=w*r[0];normal[i,1]+=w*r[2]
    out=actual.copy();out[:,1]=normal[:,0];out[:,3]=normal[:,1]
    return actual,out,{'discrete_focal_cases':len(expanded),'ordinary_mixture_weight_error':max(abs(sum(choices(r,action,size,meta).values())-1) for r in roots),'expanded_simulated_actions':more['actions'],'original_simulated_actions':info['actions']}

def pilot():
    ROOT.mkdir(exist_ok=True);q=pl.read_parquet(s.ROOT/'queries.parquet');table=q.filter(C('fold')==0)['table_id'][0]
    start=time.time();roots=s.roots_for_table(table,q.filter(C('table_id')==table),128)
    selected=[]
    for street in range(4):selected.extend([r for r in roots if r['state'].street==street][:1])
    action,size,meta=s.models(0,1);raw,rb,info=integrate(selected,action,size,meta,128)
    records=[]
    for i,r in enumerate(selected):
        for name,a,b in [('learned',0,1),('checkcall',2,3)]:
            for member in r['members']:
                own,other=r['own'],member['other'];scale=max(1.,r['pot'])
                x=(raw[i,a]-raw[i,b])[:,[own,other]]/scale;y=(rb[i,a]-rb[i,b])[:,[own,other]]/scale
                vx=x.var(0,ddof=1);vy=y.var(0,ddof=1)
                records.append({'query_id':member['query_id'],'street':r['state'].street,'arm':name,
                                'sampled_mean':x.mean(0).tolist(),'integrated_mean':y.mean(0).tolist(),
                                'sampled_per_rep_variance':vx.tolist(),'integrated_per_rep_variance':vy.tolist(),
                                'paired_mean_difference_stderr':((x-y).std(0,ddof=1)/np.sqrt(128)).tolist()})
                                                                                         
    _,prefix,_=integrate(selected,action,size,meta,16);np.testing.assert_array_equal(rb[:,:,:16],prefix)
    report={'method':__doc__,'table':table,'roots':len(selected),'replicates':128,'seconds':time.time()-start,
            'replicate_prefix_error':0,'records':records,**info,'conclusion':'pilot precision diagnostics, not evidence MAP or model selection'}
    (ROOT/'pilot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':pilot()
