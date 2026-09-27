import sys,numpy as np,polars as pl
C=pl.col; S=os.environ.get('POKEML_SCRATCH','cache')+'/'
lab=pl.read_csv('data/development_labels.csv'); pos=lab.filter(C('label')==1); partner={}
for a,b in zip(pos['player_1'],pos['player_2']): partner.setdefault(a,set()).add(b); partner.setdefault(b,set()).add(a)
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
W={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}
for arg in sys.argv[1:]:
    path,w=arg.split(':'); lo,hi=W[w]
    meta=pl.read_parquet(S+f'allpairs_{w}.parquet').select('pair_id','player_1','player_2','label')
    d=pl.read_csv(path).select('pair_id','none').join(meta,on='pair_id').join(evd.filter((C('idx')>=lo)&(C('idx')<hi)).group_by('pair_id').agg(pl.len().alias('n_ev_in')),on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0))
    p1=d['player_1'].to_list(); p2=d['player_2'].to_list(); L=d['label'].to_numpy(); nev=d['n_ev_in'].to_numpy(); r=1-d['none'].to_numpy()
    hn=np.array([(l==-1) and ((a in partner)!=(b in partner)) for a,b,l in zip(p1,p2,L)]); cl=np.array([(l==-1) and (a not in partner) and (b not in partner) for a,b,l in zip(p1,p2,L)])
    posm=(L==1)&(nev>=1); order=np.argsort(-r); rank=np.empty(len(r),int); rank[order]=np.arange(len(r)); negs=np.sort(r[L!=1]); above=len(negs)-np.searchsorted(negs,r[posm],side='right')
    print(f"{path.split('/')[-1]:24s} {w:10s} n={len(d)} pos>0.5={int((posm&(r>0.5)).sum())}/{int(posm.sum())} rec@300={np.mean(above<300):.4f} rec@1000={np.mean(above<1000):.4f} | confneg>0.5={int(((L==0)&(r>0.5)).sum())} hardneg>0.5={int((hn&(r>0.5)).sum())} top700={int((hn&(rank<700)).sum())} top1000={int((hn&(rank<1000)).sum())} | clean top700={int((cl&(rank<700)).sum())}")
