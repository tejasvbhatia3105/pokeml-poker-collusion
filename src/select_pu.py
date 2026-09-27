import json,numpy as np,pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score
names=['none','directed_transfer','soft_play','coordinated_isolation']
b=pd.read_csv('artifacts/selected_pair_oof.csv').sort_values('pair_id');p=pd.read_csv('artifacts/pu_fixed_oof.csv').sort_values('pair_id');assert b.pair_id.tolist()==p.pair_id.tolist()
y=b.truth.to_numpy();wgt=np.where(y>0,1,50);r0=1-b.none.to_numpy();r1=p.risk_score.to_numpy();family=b[names[1:]].to_numpy().argmax(1)+1;reports=[]
for w in [0,.25,.5,.75,1]:
 r=(1-w)*r0+w*r1;pa=average_precision_score(y>0,r,sample_weight=wgt);ba=np.mean([average_precision_score(y==i,r*(family==i)*(r>=.01),sample_weight=wgt) for i in range(1,4)])
 reports.append({'pu_weight':w,'weighted_pair_ap50':pa,'weighted_behavior_map50':ba,'weighted_ranking_objective':.7*pa+.1*ba,'pair_ap':average_precision_score(y>0,r)})
best=max(reports,key=lambda z:z['weighted_ranking_objective']);w=best['pu_weight']
for phase,basefile,pufile,out in [('development','artifacts/selected_pair_oof.csv','artifacts/pu_fixed_oof.csv','artifacts/pu_selected_oof.csv'),('evaluation','artifacts/selected_pair_eval.csv','artifacts/pu_fixed_eval.csv','artifacts/pu_selected_eval.csv')]:
 base=pd.read_csv(basefile).sort_values('pair_id');pu=pd.read_csv(pufile).sort_values('pair_id');assert base.pair_id.tolist()==pu.pair_id.tolist()
 risk=(1-w)*(1-base.none.to_numpy())+w*pu.risk_score.to_numpy();conditional=base[names[1:]].to_numpy();conditional=conditional/conditional.sum(axis=1,keepdims=True)
 pred=pd.DataFrame(np.column_stack([1-risk,conditional*risk[:,None]]),columns=names);pred['pair_id']=base.pair_id.to_list()
 if phase=='development':pred['truth']=base.truth.to_numpy()
 pred.to_csv(out,index=False)
 if phase=='evaluation':
  s=pd.read_csv('submission_v2.csv',keep_default_na=False).set_index('pair_id');pred=pred.set_index('pair_id').loc[s.index];s['risk_score']=1-pred.none
  s['predicted_behavior']=pred[names[1:]].idxmax(axis=1);s.loc[s.risk_score<.01,'predicted_behavior']='none';s.reset_index().to_csv('submission_v3.csv',index=False)
Path('artifacts/pu_selection.json').write_text(json.dumps({'selected':best,'candidates':reports},indent=2));print(json.dumps({'selected':best,'candidates':reports},indent=2))
