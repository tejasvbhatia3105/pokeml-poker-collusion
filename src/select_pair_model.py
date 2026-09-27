from pathlib import Path
import time,json,pandas as pd,numpy as np
from sklearn.metrics import average_precision_score
names=['none','directed_transfer','soft_play','coordinated_isolation']
while not Path('artifacts/context_eval.csv').exists():time.sleep(3)
b=pd.read_csv('artifacts/pair_oof.csv').sort_values('pair_id');c=pd.read_csv('artifacts/context_oof.csv').sort_values('pair_id');assert b.pair_id.tolist()==c.pair_id.tolist()
y=b.truth.to_numpy();weights=np.where(y>0,1,50);reports=[]
for w in [0,.25,.5,.75,1]:
 p=(1-w)*b[names].to_numpy()+w*c[names].to_numpy();r=1-p[:,0];family=p[:,1:].argmax(1)+1
 pa=average_precision_score(y>0,r,sample_weight=weights)
 ba=np.mean([average_precision_score(y==i,r*(family==i)*(r>=.01),sample_weight=weights) for i in range(1,4)])
 reports.append({'context_weight':w,'weighted_pair_ap':pa,'weighted_behavior_map':ba,'weighted_ranking_objective':.7*pa+.1*ba,'unweighted_pair_ap':average_precision_score(y>0,r)})
best=max(reports,key=lambda d:d['weighted_ranking_objective']);w=best['context_weight']
be=pd.read_csv('artifacts/pair_baseline_eval.csv').sort_values('pair_id');ce=pd.read_csv('artifacts/context_eval.csv').sort_values('pair_id');assert be.pair_id.tolist()==ce.pair_id.tolist()
p=(1-w)*be[names].to_numpy()+w*ce[names].to_numpy();pd.DataFrame(p,columns=names).assign(pair_id=be.pair_id.tolist()).to_csv('artifacts/selected_pair_eval.csv',index=False)
p=(1-w)*b[names].to_numpy()+w*c[names].to_numpy();pd.DataFrame(p,columns=names).assign(pair_id=b.pair_id.tolist(),truth=y).to_csv('artifacts/selected_pair_oof.csv',index=False)
Path('artifacts/pair_selection.json').write_text(json.dumps({'method':'Development validation; confirmed non-target weight 50 as a prevalence sensitivity analysis, not the competition metric.','candidates':reports,'selected':best},indent=2))
print(json.dumps({'selected':best,'candidates':reports},indent=2),flush=True)
