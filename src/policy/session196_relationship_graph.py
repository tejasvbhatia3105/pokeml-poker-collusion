\
\
\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,itertools
from pathlib import Path
from collections import Counter
import numpy as np,polars as pl
from scipy.special import logsumexp
from session73_exclusion_audit import metrics

ROOT=Path('artifacts/pair_session196_relationship_graph');C=pl.col

class Union:
    def __init__(self):self.p={}
    def root(self,a):
        self.p.setdefault(a,a)
        if self.p[a]!=a:self.p[a]=self.root(self.p[a])
        return self.p[a]
    def add(self,a,b):self.p[self.root(a)]=self.root(b)

def marginals(edges,p,rho):
    assert 0<rho<=1 and len(edges)<=18
    vertices=sorted({v for e in edges for v in e});vi={v:i for i,v in enumerate(vertices)}
    incidence=np.zeros((len(edges),len(vertices)),int)
    for i,(a,b) in enumerate(edges):incidence[i,vi[a]]=1;incidence[i,vi[b]]=1
    states=((np.arange(2**len(edges))[:,None]>>np.arange(len(edges)))&1).astype(float)
    degree=states@incidence;collisions=(degree*(degree-1)/2).sum(1)
    p=np.asarray(p).clip(1e-12,1-1e-12)
    logweight=states@(np.log(p)-np.log1p(-p))+collisions*np.log(rho)
    weights=np.exp(logweight-logsumexp(logweight))
    return weights@states

def check():
    edges=[('a','b'),('b','c'),('c','d')];p=np.array([.7,.8,.4]);rho=.12
    m=marginals(edges,p,rho);expected=np.zeros(3);den=0
    for state in itertools.product([0,1],repeat=3):
        degrees=Counter(v for e,k in zip(edges,state) if k for v in e)
        w=np.prod([p[i] if k else 1-p[i] for i,k in enumerate(state)])*rho**sum(v*(v-1)//2 for v in degrees.values())
        expected+=w*np.array(state);den+=w
    np.testing.assert_allclose(m,expected/den,atol=1e-14,rtol=0)
    np.testing.assert_allclose(m,marginals([e[::-1] for e in edges[::-1]],p[::-1],rho)[::-1],atol=1e-14,rtol=0)
    np.testing.assert_allclose(marginals(edges,p,1),p,atol=1e-14,rtol=0)
    np.testing.assert_allclose(marginals([edges[0]],p[:1],rho),p[:1],atol=1e-14,rtol=0)
    return dict(independent_enumeration=True,edge_and_endpoint_permutation=True,rho_one_identity=True,singleton_identity=True)

def main():
    ROOT.mkdir(exist_ok=True);proof=check()
    labs=pl.read_csv('data/development_labels.csv').join(pl.read_parquet('artifacts/evidence_session115_relationship_data/bags.parquet').select('pair_id','fold','table_id'),on='pair_id',validate='1:1')
    rho={};prior=[]
    for f in range(4):
        tr=labs.filter((C('label')==1)&(C('fold')!=f));counts=Counter(tr['player_1'].to_list()+tr['player_2'].to_list())
        multi=sum(v>1 for v in counts.values());rho[f]=(multi+.5)/(len(counts)+1)
        prior.append(dict(excluded_fold=f,training_pairs=tr['pair_id'].to_list(),target_players=len(counts),multi_partner_players=multi,rho=rho[f]))
    reports=[]
    for window in ['full','first_2000','last_2000']:
        q=pl.read_parquet(f'artifacts/pair_session75_missing_support/{window}.parquet',columns=['pair_id','player_1','player_2','risk_score','label','n_truth'])
        allgraph=Union()
        for a,b in q.select('player_1','player_2').iter_rows():allgraph.add(a,b)
        foldmap={};tablemap={}
        for a,b,f,t in labs.select('player_1','player_2','fold','table_id').iter_rows():
            r=allgraph.root(a);assert r==allgraph.root(b)
            assert r not in foldmap or foldmap[r]==f
            foldmap[r]=f;tablemap[r]=t
        active=np.flatnonzero(q['risk_score'].to_numpy()>=.05)
        sub=q[active];graph=Union()
        for a,b in sub.select('player_1','player_2').iter_rows():graph.add(a,b)
        groups={}
        for i,a in enumerate(sub['player_1']):groups.setdefault(graph.root(a),[]).append(i)
        scores=q['risk_score'].to_numpy().copy();records=[]
        for ix in groups.values():
            g=sub[ix];rr=allgraph.root(g['player_1'][0]);f=foldmap.get(rr)
            rate=rho[f] if f is not None else .1
            edges=list(g.select('player_1','player_2').iter_rows())
            original=g['risk_score'].to_numpy()
            revised=original.copy() if len(ix)==1 else marginals(edges,original,rate)
            assert np.all(revised<=original+1e-10) and np.isfinite(revised).all()
            scores[active[ix]]=revised
            records.append(dict(edges=len(ix),fold=f,rho=rate))
        q=q.with_columns(pl.Series('graph_score',scores),pl.Series('pool',[allgraph.root(a) for a in q['player_1']]))
        q.write_parquet(ROOT/f'{window}.parquet')
        row=dict(window=window,baseline=metrics(q,'risk_score'),graph=metrics(q,'graph_score'),
                 graph_components=len(groups),largest_component_edges=max(r['edges'] for r in records),
                 active_edges=len(active),unmapped_active_components=sum(r['fold'] is None for r in records),
                 changed_pairs=int((abs(scores-q['risk_score'].to_numpy())>1e-12).sum()))
                                                                                        
        from sklearn.metrics import average_precision_score
        known=q.filter(C('label')>=0);pools=sorted(known['pool'].unique());index={p:i for i,p in enumerate(pools)}
        membership=np.array([index[p] for p in known['pool']]);rng=np.random.default_rng(196)
        y=known['label'].to_numpy();base=known['risk_score'].to_numpy();new=known['graph_score'].to_numpy();boot=[]
        for _ in range(1000):
            counts=np.bincount(rng.integers(0,len(pools),len(pools)),minlength=len(pools));w=counts[membership]*np.where(y,1,50)
            boot.append(average_precision_score(y,new,sample_weight=w)-average_precision_score(y,base,sample_weight=w))
        row['weight50_AP_delta_CI95']=np.quantile(boot,[.025,.975]).tolist();reports.append(row)
        print(json.dumps(row),flush=True)
    (ROOT/'report.json').write_text(json.dumps(dict(method=__doc__,math_checks=proof,priors=prior,windows=reports),indent=2))

if __name__=='__main__':main()
