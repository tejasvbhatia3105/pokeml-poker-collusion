import os
import polars as pl, numpy as np, itertools, json
C=pl.col; S=os.environ.get('POKEML_SCRATCH','cache')+'/'
lab=pl.read_csv('data/development_labels.csv'); pos=lab.filter(C('label')==1); partner={}
for a,b in zip(pos['player_1'],pos['player_2']): partner.setdefault(a,set()).add(b); partner.setdefault(b,set()).add(a)
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
W={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}
V=['seq_v6','seq_v6s1','seq_v6s2','seq_v7','seq_v7s1','seq_v7s2','seq_v8','seq_v8s1','seq_v8s2']
g=pl.read_csv('artifacts/candidate_r4s/pair_oof_allpairs.csv').select('pair_id',(1-C('none')).alias('gbdt'))
data={}
for w,(lo,hi) in W.items():
    meta=pl.read_parquet(S+f'allpairs_{w}.parquet').select('pair_id','player_1','player_2','label')
    d=meta.join(evd.filter((C('idx')>=lo)&(C('idx')<hi)).group_by('pair_id').agg(pl.len().alias('n_ev_in')),on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0)).join(g,on='pair_id',how='left')
    for v in V: d=d.join(pl.read_csv(f'artifacts/{v}/dev_{w}_lpo.csv').select('pair_id',(1-C('none')).alias(v)),on='pair_id',how='left')
    d=d.drop_nulls()
    p1=d['player_1'].to_list(); p2=d['player_2'].to_list(); L=d['label'].to_numpy(); nev=d['n_ev_in'].to_numpy()
    hn=np.array([(l==-1) and ((a in partner)!=(b in partner)) for a,b,l in zip(p1,p2,L)]); posm=(L==1)&(nev>=1)
    data[w]=dict(L=L,hn=hn,posm=posm,R={k:np.log(np.clip(d[k].to_numpy(),1e-9,1)) for k in V+['gbdt']})
def metrics(w,lr):
    D=data[w]; r=lr; L=D['L']; negs=np.sort(r[L!=1]); above=len(negs)-np.searchsorted(negs,r[D['posm']],side='right')
    order=np.argsort(-r); rank=np.empty(len(r),int); rank[order]=np.arange(len(r))
    return dict(rec300=float(np.mean(above<300)),rec1000=float(np.mean(above<1000)),hn700=int((D['hn']&(rank<700)).sum()),hn1000=int((D['hn']&(rank<1000)).sum()))
def blend(w,wg,w6,w7,w8,seeds=True):
    R=data[w]['R']; g6=np.mean([R[k] for k in (V[0:3] if seeds else V[0:1])],0); g7=np.mean([R[k] for k in (V[3:6] if seeds else V[3:4])],0); g8=np.mean([R[k] for k in (V[6:9] if seeds else V[6:7])],0)
    seq=(w6*g6+w7*g7+w8*g8)/(w6+w7+w8); return (wg*R['gbdt']+seq)/(wg+1)
rows=[]
for wg,w6,w7,w8,sd in itertools.product([0.5,0.75,1.0,1.5],[1.0],[0.5,1.0,2.0],[0.0,0.25,0.5,1.0],[True,False]):
    m={w:metrics(w,blend(w,wg,w6,w7,w8,sd)) for w in W}
    rows.append(dict(wg=wg,w6=w6,w7=w7,w8=w8,seeds=sd,rec300=np.mean([m[w]['rec300'] for w in W]),rec1000=np.mean([m[w]['rec1000'] for w in W]),hn700=sum(m[w]['hn700'] for w in W),hn1000=sum(m[w]['hn1000'] for w in W),**{f'{w}_r300':round(m[w]['rec300'],4) for w in W}))
df=pl.DataFrame(rows).with_columns((C('rec300')+C('rec1000')-C('hn1000')/3000).alias('obj')).sort('obj',descending=True)
pl.Config.set_tbl_rows(14); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(200)
print(df.head(12)); print('R26-equivalent (wg=1,w6=1,w7=1,w8=0.5,seeds):'); print(df.filter((C('wg')==1.0)&(C('w7')==1.0)&(C('w8')==0.5)&(C('seeds')==True)))
df.write_csv(S+'blend_grid.csv')
