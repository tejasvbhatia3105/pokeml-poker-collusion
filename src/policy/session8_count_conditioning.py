\
\
\
\
\
import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from session8_data import ROOT,C,reference
from session6_priority import inclusion
def categorical(a,b):
    scale=np.maximum(1,a+b);return np.column_stack([a/scale,b/scale])
def conditioned(p,k):
                                                                          
                                                                             
    p=np.asarray(p,dtype=np.float64)
    p=p/np.maximum(1,p.sum(1))[:,None]
    a,b=p.T;ordinary=inclusion(a,b);s=a+b
    def scan(v):
        z=np.zeros((len(v)+1,k));z[0,0]=1
        for i,pr in enumerate(v):z[i+1]=np.r_[z[i,0]*(1-pr),z[i,1:]*(1-pr)+z[i,:-1]*pr]
        return z
    prefix=scan(s);suffix=scan(s[::-1])[::-1];den=1-prefix[-1].sum()
    if den<1e-12:return ordinary
    subtract=np.array([s[i]*np.convolve(prefix[i],suffix[i+1])[:k-1].sum() for i in range(len(s))]);out=(ordinary-subtract)/den
    assert out.min()>-1e-8 and out.max()<1+1e-8
    return np.clip(out,0,1)
def verify():
    count=0
    for n in [3,4,5,6]:
        p=np.random.default_rng(n).dirichlet([5,1,1],n)[:,1:]
        for k in [3,5]:
            if n<k:continue
            acc=np.zeros(n);den=0
            for cats in itertools.product(range(3),repeat=n):
                prob=np.prod([1-p[i].sum() if c==0 else p[i,c-1] for i,c in enumerate(cats)]);events=[i for i,c in enumerate(cats) if c==1]+[i for i,c in enumerate(cats) if c==2]
                if len(events)>=k:acc[events[:5]]+=prob;den+=prob
                count+=1
            assert np.allclose(conditioned(p,k),acc/den,atol=1e-10)
    return count
def main():
    checked=verify();d=reference()
    for name,path in [('cat','artifacts/evidence_session6/priority_ordered_oof.parquet'),('hist','artifacts/evidence_session7/hist_events_oof.parquet')]:
        z=pl.read_parquet(path);d=d.join(z.select('pair_id','hand_id',C('primary').alias(name+'_a'),C('secondary').alias(name+'_b')),on=['pair_id','hand_id'],validate='1:1')
    rows=[];allparts=[]
    for (pid,),g in d.group_by('pair_id'):
        g=g.sort('time','hand_id');cat=categorical(g['cat_a'].to_numpy(),g['cat_b'].to_numpy());hist=categorical(g['hist_a'].to_numpy(),g['hist_b'].to_numpy());h=(cat+hist)/2;base=g['r27_base'].to_numpy();unc=.25*base+.25*inclusion(*cat.T)+.5*inclusion(*h.T);assert np.allclose(unc,g['r29'].to_numpy(),atol=1e-10)
        g=g.with_columns(*[pl.Series('minimum_'+str(k),.25*base+.25*conditioned(cat,k)+.5*conditioned(h,k)) for k in [3,5]])
        truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth));row={'pair_id':pid,'table_id':g['table_id'][0],'family':g['behavior_family'][0],'fold':g['fold'][0]}
        for name in ['r29','minimum_3','minimum_5']:
            hand=g.sort([name,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([x in truth for x in hand]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
        rows.append(row);allparts.append(g.select('pair_id','hand_id','r29','minimum_3','minimum_5'))
    r=pl.DataFrame(rows);r.write_csv(ROOT/'count_conditioning.csv');pl.concat(allparts).write_parquet(ROOT/'count_conditioning_predictions.parquet');names=['r29','minimum_3','minimum_5'];report={'enumerated_paths_checked':checked,'overall':r.select(C(names).mean()).to_dicts(),'families':r.group_by('family').agg(C(names).mean()).to_dicts(),'folds':r.group_by('fold').agg(C(names).mean()).sort('fold').to_dicts()};(ROOT/'count_conditioning.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
