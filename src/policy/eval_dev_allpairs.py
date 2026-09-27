\
import sys,numpy as np,polars as pl
from collections import Counter
C=pl.col
lab=pl.read_csv('data/development_labels.csv'); pos=lab.filter(C('label')==1); partner={}
for a,b in zip(pos['player_1'],pos['player_2']): partner.setdefault(a,set()).add(b); partner.setdefault(b,set()).add(a)
S='cache/'
meta=pl.read_parquet(S+'allpairs_full.parquet').select('pair_id','player_1','player_2','label')
for path in sys.argv[1:]:
    d=pl.read_csv(path).with_columns((1-C('none')).alias('risk')).join(meta,on='pair_id')
    p1=d['player_1'].to_list(); p2=d['player_2'].to_list(); L=d['label'].to_numpy(); r=d['risk'].to_numpy()
    hn=np.array([(l==-1) and ((a in partner)!=(b in partner)) for a,b,l in zip(p1,p2,L)]); cl=np.array([(l==-1) and (a not in partner) and (b not in partner) for a,b,l in zip(p1,p2,L)])
    dbl=np.array([(l==1) and (len(partner.get(a,()))>1 or len(partner.get(b,()))>1) for a,b,l in zip(p1,p2,L)])
    order=np.argsort(-r); rank=np.empty(len(r),int); rank[order]=np.arange(len(r))
    print(f"{path}: labpos>0.5 {int(((L==1)&(r>0.5)).sum())}/372, double-partner pos>0.5 {int((dbl&(r>0.5)).sum())}/{int(dbl.sum())} | confneg>0.5 {int(((L==0)&(r>0.5)).sum())} | hardneg>0.5 {int((hn&(r>0.5)).sum())} >0.9 {int((hn&(r>0.9)).sum())} top700 {int((hn&(rank<700)).sum())} top1000 {int((hn&(rank<1000)).sum())} | clean>0.5 {int((cl&(r>0.5)).sum())} top700 {int((cl&(rank<700)).sum())} | labpos top700 {int(((L==1)&(rank<700)).sum())}")
    if 'risk_new' in d.columns:
        ch=(d['risk_new']!=d['risk_orig']).to_numpy()&(d['n_kept']>0).to_numpy(); print(f"   rescored pairs {int(ch.sum())}: hardneg among them {int((hn&ch).sum())}, labelled pos {int(((L==1)&ch).sum())}, clean {int((cl&ch).sum())}; hardneg dropped from >0.5: {int((hn&ch&(d['risk_orig'].to_numpy()>0.5)&(r<=0.5)).sum())}, labelled pos dropped: {int(((L==1)&ch&(d['risk_orig'].to_numpy()>0.5)&(r<=0.5)).sum())}, clean dropped: {int((cl&ch&(d['risk_orig'].to_numpy()>0.5)&(r<=0.5)).sum())}")
