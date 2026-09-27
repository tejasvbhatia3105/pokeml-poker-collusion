\
\
\
import sys,numpy as np,polars as pl
import os
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
lab=pl.read_csv('data/development_labels.csv'); pos=lab.filter(C('label')==1); partner={}
for a,b in zip(pos['player_1'],pos['player_2']): partner.setdefault(a,set()).add(b); partner.setdefault(b,set()).add(a)
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
W={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}
specs={}
for part in sys.argv[1].split(';'):
    name,groups=part.split('='); specs[name]=[(g.split(':')[0].split(','),float(g.split(':')[1]) if ':' in g else 1.0) for g in groups.split('|')]
models=sorted({m for gs in specs.values() for g,_ in gs for m in g})
g=pl.read_csv('artifacts/candidate_r4s/pair_oof_allpairs.csv').select('pair_id',(1-C('none')).alias('gbdt'))
def metrics(r,L,posm,hn):
    negs=np.sort(r[L!=1]); above=len(negs)-np.searchsorted(negs,r[posm],side='right')
    order=np.argsort(-r); rank=np.empty(len(r),int); rank[order]=np.arange(len(r))
    return dict(rec300=np.mean(above<300),rec1000=np.mean(above<1000),pos5=int((posm&(r>0.5)).sum()),npos=int(posm.sum()),hn5=int((hn&(r>0.5)).sum()),hn700=int((hn&(rank<700)).sum()),hn1000=int((hn&(rank<1000)).sum()))
out={n:{} for n in specs}
for w,(lo,hi) in W.items():
    meta=pl.read_parquet(S+f'allpairs_{w}.parquet').select('pair_id','player_1','player_2','label')
    d=meta.join(evd.filter((C('idx')>=lo)&(C('idx')<hi)).group_by('pair_id').agg(pl.len().alias('n_ev_in')),on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0)).join(g,on='pair_id',how='left')
    for v in models: d=d.join(pl.read_csv(f'artifacts/{v}/dev_{w}_lpo.csv').select('pair_id',(1-C('none')).alias(v)),on='pair_id',how='left')
    d=d.drop_nulls(); R={k:np.log(np.clip(d[k].to_numpy(),1e-9,1)) for k in models+['gbdt']}
    p1=d['player_1'].to_list(); p2=d['player_2'].to_list(); L=d['label'].to_numpy(); nev=d['n_ev_in'].to_numpy()
    hn=np.array([(l==-1) and ((a in partner)!=(b in partner)) for a,b,l in zip(p1,p2,L)]); posm=(L==1)&(nev>=1)
    for n,gs in specs.items():
        seq=sum(wt*np.mean([R[m] for m in grp],0) for grp,wt in gs)/sum(wt for _,wt in gs)
        out[n][w]=metrics(np.exp(0.5*(R['gbdt']+seq)),L,posm,hn)
for n in specs:
    m=out[n]; print(f"{n:18s} rec300 {' '.join(f'{m[w]['rec300']:.4f}' for w in W)} | rec1000 {' '.join(f'{m[w]['rec1000']:.4f}' for w in W)} | pos>0.5 {' '.join(str(m[w]['pos5']) for w in W)} | hn>0.5 {' '.join(str(m[w]['hn5']) for w in W)} | hn1000 {' '.join(str(m[w]['hn1000']) for w in W)} | mean rec300 {np.mean([m[w]['rec300'] for w in W]):.4f}")
