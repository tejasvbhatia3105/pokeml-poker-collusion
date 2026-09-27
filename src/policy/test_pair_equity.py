import numpy as np
from cards import rank,CARD
from pair_equity import pair_equity
rng=np.random.default_rng(218);states=np.full((600,9),-1,np.int8)
for i in range(600):
 nb=[0,3,4,5][i%4];states[i,:4+nb]=rng.choice(52,size=4+nb,replace=False)
p=pair_equity(states);swap=states.copy();swap[:,:2]=states[:,2:4];swap[:,2:4]=states[:,:2];assert np.max(np.abs(p+pair_equity(swap)-1))<1e-6
for i in range(600):
 c=states[i];nb=(c[4:]>=0).sum()
 if nb<4:continue
 tail=[None] if nb==5 else [v for v in range(52) if v not in c];scores=[]
 for v in tail:
  board=c[4:4+nb].tolist()+([] if v is None else [v]);a=rank(c[:2].tolist()+board);b=rank(c[2:4].tolist()+board);scores.append(1 if a>b else .5 if a==b else 0)
 assert abs(p[i]-np.mean(scores))<1e-6
print('PASS: 600 swap symmetry cases and 300 exact turn/river reference comparisons')
