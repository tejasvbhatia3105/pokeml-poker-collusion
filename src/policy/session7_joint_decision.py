\
\
\
\
\
\
import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from session7_compare import frame
ROOT=Path('artifacts/evidence_session7');C=pl.col
def optimize(marginal,joint,hand,k=5):
    n=len(marginal);dp={0:0.};parent={};diagonal=np.diag(joint)
    for size in range(1,min(k,n)+1):
        for ix in itertools.combinations(range(n),size):
            bits=[1<<i for i in ix];mask=sum(bits);values=[dp[mask^bit]+(marginal[i]+joint[i,list(ix)].sum()-diagonal[i])/size for i,bit in zip(ix,bits)]
            best=int(np.argmax(values));dp[mask]=values[best];parent[mask]=ix[best]
    chosen=max((mask for mask in dp if mask.bit_count()==min(k,n)),key=lambda mask:dp[mask]);answer=[]
    while chosen:
        i=parent[chosen];answer.append(i);chosen^=1<<i
    return [hand[i] for i in reversed(answer)]
def main():
    d=frame().join(pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet').select('pair_id','hand_id','primary','secondary'),on=['pair_id','hand_id'],validate='1:1');rows=[]
    with threadpool_limits(limits=2):
        for (pid,),g in d.sort('pair_id','time','hand_id').group_by('pair_id',maintain_order=True):
            p1=g['primary'].to_numpy();p2=g['secondary'].to_numpy();scale=np.maximum(1,p1+p2);a=p1/scale;b=p2/scale;hand=g['hand_id'].to_numpy();base=g['r27_base'].to_numpy();mix=g['r28'].to_numpy();idx=np.lexsort((hand,-mix))[:12];true=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(true))
            def ap(chosen):
                y=np.array([h in true for h in chosen]);return float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
            row={'pair_id':pid,'table_id':g['table_id'][0],'family':g['behavior_family'][0],'fold':g['fold'][0],'r28':ap(hand[idx[:5]])}
            for seed in (71,72):
                uniform=qmc.Sobol(d=len(g),scramble=True,seed=seed).random_base2(13);primary=uniform<a;secondary=(uniform>=a)&(uniform<a+b)
                listed=(primary&(np.cumsum(primary,axis=1)<=5))|(secondary&(np.cumsum(secondary,axis=1)<=np.maximum(0,5-primary.sum(1))[:,None]))
                count=np.maximum(1,listed.sum(1));y=listed[:,idx].astype(float);weighted=y/count[:,None];marg=.5*weighted.mean(0)+.5*base[idx]/5;joint=.5*(y.T@weighted)/len(y)+.5*np.outer(base[idx],base[idx])/5
                np.fill_diagonal(joint,marg);chosen=optimize(marg,joint,hand[idx]);row[f'joint_seed{seed}']=ap(chosen);row[f'hands_seed{seed}']=' '.join(chosen)
            rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'joint_decision.csv');names=['r28','joint_seed71','joint_seed72'];pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n'));ix=np.random.default_rng(712).integers(0,len(pool),(3000,len(pool)));report={}
    for name in names:
        delta=pool[name].to_numpy()-pool['r28'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report[name]={'map5':r[name].mean(),'gain':r[name].mean()-r['r28'].mean(),'ci95':np.quantile(boot,[.025,.975]).tolist()}
    report['sampling_changed_rankings']=int((r['hands_seed71']!=r['hands_seed72']).sum());(ROOT/'joint_decision.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
