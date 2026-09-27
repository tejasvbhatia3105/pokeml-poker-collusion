\
\
\
\
\
import os,json,itertools,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from session11_build_candidate import models,features,NAMES
from session8_data import hand_data
from session7_joint_decision import optimize
ROOT=Path('artifacts/evidence_session12');C=pl.col
CONFIG={'draw_power':13,'shortlist':12,'scramble_seeds':[121,122],'r27_joint':'independent Bernoulli approximation, denominator five','conditional_sampling':'exact sequential count-conditioned categorical probabilities','objective':'expected AP, not marginal inclusion sorting'}
def tails(p,k):
    s=p.sum(1);out=np.zeros((len(p)+1,k+1));out[:,0]=1
    for i in range(len(p)-1,-1,-1):out[i,1:]=(1-s[i])*out[i+1,1:]+s[i]*out[i+1,:-1]
    return out
def listed_samples(p,u,k=0):
    p=np.asarray(p,np.float64);p=p/np.maximum(1,p.sum(1))[:,None];n=len(p);N=len(u);a=np.zeros((N,n),bool);b=a.copy();count=np.zeros(N,int);tail=tails(p,k) if k else None
    for i in range(n):
        if k:
            need=np.maximum(0,k-count);num=p[i].sum()*tail[i+1,np.maximum(0,need-1)];den=tail[i,need];assert np.all(den>0);event=np.clip(num/den,0,1);pa=event*p[i,0]/max(p[i].sum(),1e-30);pb=event-pa
        else:pa,pb=p[i]
        a[:,i]=u[:,i]<pa;b[:,i]=(u[:,i]>=pa)&(u[:,i]<pa+pb);count+=a[:,i]|b[:,i]
    assert np.all(count>=k)
    chosen=(a&(a.cumsum(1)<=5))|(b&(b.cumsum(1)<=np.maximum(0,5-a.sum(1))[:,None]))
    return chosen
def check():
    p=np.random.default_rng(12).dirichlet([3,1,1],6)[:,1:];error=0.;paths=0
    for k in [3,5]:
        tail=tails(p,k)
        for cat in itertools.product(range(3),repeat=6):
            if sum(v>0 for v in cat)<k:continue
            mass=np.prod([1-p[i].sum() if c==0 else p[i,c-1] for i,c in enumerate(cat)])/tail[0,k];conditional=1.;count=0
            for i,c in enumerate(cat):
                need=max(0,k-count);event=p[i].sum()*tail[i+1,max(0,need-1)]/tail[i,need];cp=1-event if c==0 else event*p[i,c-1]/p[i].sum();conditional*=cp;count+=c>0
            error=max(error,abs(mass-conditional));paths+=1
    assert error<1e-12
    return {'enumerated_feasible_paths':paths,'conditional_path_probability_error':error}
def moments(listed,idx):
    y=listed[:,idx].astype(np.float64);weighted=y/np.maximum(1,listed.sum(1))[:,None]
    return weighted.mean(0),y.T@weighted/len(y)
def main():
    (ROOT/'joint_decision_config.json').write_text(json.dumps(CONFIG,indent=2));checks=check();(ROOT/'joint_decision_checks.json').write_text(json.dumps(checks,indent=2));m=models('conditional_family');d=hand_data();route=pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score');r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id',C('conditional_family').alias('r30'));parts=[];rows=[];start=time.time();done=0
    with threadpool_limits(limits=2):
        for f in range(4):
            raw=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time');q=d.filter(C('fold')==f).join(raw,on=['pair_id','hand_id'],validate='1:1').join(route,on='pair_id',validate='m:1').join(r30,on=['pair_id','hand_id'],validate='1:1')
            for (pid,),g in q.group_by('pair_id'):
                g=g.sort('time','hand_id');hand=g['hand_id'].to_numpy();base=g['base'].to_numpy();cat=g.select('cat_primary','cat_secondary').to_numpy();cat=cat/np.maximum(1,cat.sum(1))[:,None];idx=np.lexsort((hand,-g['r30'].to_numpy()))[:CONFIG['shortlist']];out=g.select('pair_id','hand_id','r30');truth=set(g.filter(C('evidence')==1)['hand_id']);row={'pair_id':pid,'table_id':g['table_id'][0],'fold':f,'family':g['behavior_family'][0]};x,prior=features(g);probs=[]
                for net,mu,sd,minimums in m[f]:
                    xx=torch.tensor(np.clip((x-mu)/sd,-6,6))[None];mask=torch.ones((1,len(g)),dtype=torch.bool)
                    with torch.no_grad():logits=torch.tensor(prior)+net(xx,mask)[0];p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:1]),logits],1),1).numpy()[:,1:];probs.append((p,minimums[g['behavior_family'][0]]))
                for seed in CONFIG['scramble_seeds']:
                    name=f'joint_{seed}'
                    if g['risk_score'][0]<.05:chosen=g.sort('r30','hand_id',descending=[True,False])['hand_id'].to_list()[:5]
                    else:
                        u=qmc.Sobol(d=len(g),scramble=True,seed=seed).random_base2(CONFIG['draw_power']);ma=.25*base[idx]/5;joint=.25*np.outer(base[idx],base[idx])/5
                        for pp,k,weight in [(cat,0,.25)]+[(p,k,.25) for p,k in probs]:
                            mm,jj=moments(listed_samples(pp,u,k),idx);ma+=weight*mm;joint+=weight*jj
                        np.fill_diagonal(joint,ma);chosen=optimize(ma,joint,hand[idx])
                    positions={h:5-j for j,h in enumerate(chosen)};out=out.with_columns(pl.Series(name,[positions.get(h,0) for h in hand]));v=np.array([h in truth for h in chosen]);row[name]=float((v*v.cumsum()/np.arange(1,len(v)+1)).sum()/min(5,len(truth)));row[name+'_hands']=' '.join(chosen)
                rows.append(row);parts.append(out);done+=1
                if done%50==0:print('joint R30',done,'seconds',round(time.time()-start,1),flush=True)
    pl.concat(parts).write_parquet(ROOT/'joint_decision_oof.parquet');pl.DataFrame(rows).write_csv(ROOT/'joint_decision_pairs.csv');print('joint complete',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
