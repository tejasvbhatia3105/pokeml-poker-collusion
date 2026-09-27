import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
root=Path('artifacts/policy');names=['none','directed_transfer','soft_play','coordinated_isolation']
while not (root/'combined_oof.csv').exists():time.sleep(3)
a=pd.read_csv('artifacts/selected_pair_oof.csv').sort_values('pair_id');b=pd.read_csv(root/'residual_oof.csv').sort_values('pair_id');c=pd.read_csv(root/'combined_oof.csv').sort_values('pair_id');assert a.pair_id.tolist()==b.pair_id.tolist()==c.pair_id.tolist();y=b.truth.to_numpy();weights=np.where(y>0,1,50);rows=[]
for rw,cw,ow in [(1,0,0),(0,1,0),(.5,.5,0),(.75,0,.25),(.5,.25,.25),(0,.75,.25)]:
 p=rw*b[names].to_numpy()+cw*c[names].to_numpy()+ow*a[names].to_numpy();risk=1-p[:,0];pred=p[:,1:].argmax(1)+1
 ap=average_precision_score(y>0,risk,sample_weight=weights);bm=np.mean([average_precision_score(y==k,risk*(pred==k),sample_weight=weights) for k in [1,2,3]])
 rows.append({'residual':rw,'combined':cw,'v2':ow,'weighted_ap':ap,'weighted_behavior':bm,'selection_score':.7*ap+.1*bm})
best=max(rows,key=lambda r:r['selection_score']);p=best['residual']*b[names].to_numpy()+best['combined']*c[names].to_numpy()+best['v2']*a[names].to_numpy()
                                                                                         
nov=pd.read_csv(root/'novelty_oof.csv').sort_values('pair_id');folds=json.loads(Path('artifacts/folds.json').read_text());pn=np.zeros(len(y));cal=[]
for f in folds:
 va=nov.table_id.isin(f['valid_tables']).to_numpy();tr=~va;m=LogisticRegression(C=1).fit(nov.novelty.to_numpy()[tr,None],y[tr]>0);pn[va]=m.predict_proba(nov.novelty.to_numpy()[va,None])[:,1];cal.append(m)
novrows=[]
for floor in [0,.25,.5,.75]:
 risk=np.maximum(1-p[:,0],floor*pn)
 novrows.append({'floor':floor,'weighted_ap':average_precision_score(y>0,risk,sample_weight=weights),'raised_pairs':int((floor*pn>1-p[:,0]).sum())})
report={'mixtures':rows,'selected':best,'novelty_floor_audit':novrows};(root/'pair_selection.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
pd.DataFrame(p,columns=names).assign(pair_id=b.pair_id.to_numpy(),truth=y).to_csv(root/'selected_oof.csv',index=False)
while not (root/'combined_eval.csv').exists():time.sleep(3)
ae=pd.read_csv('artifacts/selected_pair_eval.csv').sort_values('pair_id');be=pd.read_csv(root/'residual_eval.csv').sort_values('pair_id');ce=pd.read_csv(root/'combined_eval.csv').sort_values('pair_id');assert ae.pair_id.tolist()==be.pair_id.tolist()==ce.pair_id.tolist()
pe=best['residual']*be[names].to_numpy()+best['combined']*ce[names].to_numpy()+best['v2']*ae[names].to_numpy();pd.DataFrame(pe,columns=names).assign(pair_id=be.pair_id.to_numpy()).to_csv(root/'selected_eval.csv',index=False)
