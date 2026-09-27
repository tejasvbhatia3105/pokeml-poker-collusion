from collections import Counter
from itertools import combinations
import numpy as np,time
from cards import rank,features,preflop_table,CARD

def rank5(c):
 r=sorted([v//4+2 for v in c],reverse=True);counts=Counter(r);groups=sorted(((n,v) for v,n in counts.items()),reverse=True);flush=len({v%4 for v in c})==1;u=sorted(set(r));straight=max(u) if len(u)==5 and u[-1]-u[0]==4 else (5 if u==[2,3,4,5,14] else 0)
 if flush and straight:cat,k=8,[straight]
 elif groups[0][0]==4:cat,k=7,[groups[0][1],groups[1][1]]
 elif [v[0] for v in groups]==[3,2]:cat,k=6,[groups[0][1],groups[1][1]]
 elif flush:cat,k=5,r
 elif straight:cat,k=4,[straight]
 elif groups[0][0]==3:cat,k=3,[groups[0][1]]+sorted([v for n,v in groups[1:]],reverse=True)
 elif [v[0] for v in groups[:2]]==[2,2]:cat,k=2,sorted([groups[0][1],groups[1][1]],reverse=True)+[groups[2][1]]
 elif groups[0][0]==2:cat,k=1,[groups[0][1]]+sorted([v for n,v in groups[1:]],reverse=True)
 else:cat,k=0,r
 return (cat<<24)+sum(v<<(20-4*i) for i,v in enumerate(k))
rng=np.random.default_rng(7);t=time.time()
for n in [5,6,7]:
 for _ in range(2500):
  cards=rng.choice(52,n,replace=False).tolist();assert rank(cards)==max(rank5(c) for c in combinations(cards,5)),cards
print('7500 random exact-rank comparisons passed',round(time.time()-t,2),flush=True)
t=time.time();p=preflop_table();assert p[12,12,0]>.8 and p[5,0,0]<.4
print('AA equity',p[12,12,0],'72o equity',p[5,0,0],'preflop seconds',round(time.time()-t,2),flush=True)
s=np.stack([rng.choice(52,7,replace=False) for _ in range(10000)]);s[:,5:]=-1;t=time.time();v=features(s,96);assert ((v[:,2]>=0)&(v[:,2]<=1)).all()
print('10000 flop states seconds',round(time.time()-t,2),flush=True)
