import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,polars as pl,pandas as pd
root=Path('artifacts/policy');C=pl.col;labels=pl.read_csv('data/development_labels.csv').filter(C('label')==1).select('pair_id','behavior_family');pairs=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(C('phase')=='development').join(labels.lazy(),on='pair_id').collect();h=pl.scan_parquet(str(root/'hand_features/*.parquet')).filter(C('phase')=='development').join(labels.lazy().select('pair_id'),on='pair_id').collect().sort('pair_id','time_index');folds=json.loads((root/'table_folds.json').read_text());params=[np.load(root/f'open_set_floor/calibrator_fold{f}.npz') for f in range(4)];truth={}
for p,hand in pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').iter_rows():truth.setdefault(p,set()).add(hand)
metadata={r['pair_id']:r for r in pairs.to_dicts()};results=[];selections=[]
for key,z in h.group_by('pair_id',maintain_order=True):
 pid=key[0];r=metadata[pid];par=params[folds[r['table_id']]];cols=par['columns'].tolist();v=np.array([r[c] for c in cols]);delta=(v-par['med'])/par['scale'];importance=np.log1p(np.abs(delta));top=np.argsort(importance)[-3:];score=np.zeros(len(z))
 for i in top:
  name=cols[i];channel=name[:-2] if name.endswith('_z') else name.rsplit('_w10_',1)[0];direction=np.sign(delta[i]);rr=z[channel+'_r'].to_numpy();vv=z[channel+'_v'].to_numpy();contrib=direction*rr/np.sqrt(vv+1);score+=importance[i]*np.maximum(contrib,0)
 order=np.argsort(-score,kind='stable')[:5];hands=z['hand_id'].to_numpy()[order].tolist();rel=np.array([hand in truth[pid] for hand in hands]);map5=float((np.cumsum(rel)/np.arange(1,len(rel)+1)*rel).sum()/min(5,len(truth[pid])));results.append({'pair_id':pid,'behavior':r['behavior_family'],'map5':map5});selections.append({'pair_id':pid,'hands':hands})
pd.DataFrame(results).to_csv(root/'open_set_floor/generic_evidence_validation.csv',index=False);pl.DataFrame(selections).write_parquet(root/'open_set_floor/generic_evidence_selection.parquet');print(pd.DataFrame(results).groupby('behavior').map5.mean().to_string());print('overall',pd.DataFrame(results).map5.mean())
