\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier
from evidence_data import load
root=Path('artifacts/policy');dest=root/'unseen_evidence';dest.mkdir(exist_ok=True);d,cols=load();oldcols=json.loads(Path('artifacts/rank_columns.json').read_text());families=['directed_transfer','soft_play','coordinated_isolation'];folds=json.loads(Path('artifacts/folds.json').read_text());parts=[];t=time.time()
X=d.select(cols).to_numpy();y=d['evidence'].to_numpy();family=d['behavior_family'].to_numpy();g=d['table_id'].to_numpy()
for target in families:
    for f in folds:
        va=np.isin(g,f['valid_tables'])&(family==target);tr=~np.isin(g,f['valid_tables'])&(family!=target)
        m=CatBoostClassifier(iterations=600,depth=5,learning_rate=.035,loss_function='Logloss',l2_leaf_reg=8,thread_count=4,random_seed=921+f['fold'],verbose=False,allow_writing_files=False)
        m.fit(X[tr],y[tr]);generic=m.predict_proba(X[va])[:,1];m.save_model(str(dest/f'without_{target}_fold{f["fold"]}.cbm'))
        z=d.filter(pl.Series(va));available=[]
        for other in families:
            if other==target:continue
            old=CatBoostClassifier();old.load_model(f'artifacts/rank_{other}_fold{f["fold"]}.cbm');seq=CatBoostClassifier();seq.load_model(str(root/f'sequence/evidence_{other}_fold{f["fold"]}.cbm'))
            available.append(.25*old.predict_proba(z.select(oldcols).to_numpy())[:,1]+.75*seq.predict_proba(X[va])[:,1])
        parts.append(z.select('pair_id','hand_id','behavior_family','evidence').with_columns(pl.Series('generic',generic),pl.Series('available_family_max',np.max(available,axis=0))))
        print(target,f['fold'],round(time.time()-t,1),flush=True)
result=pl.concat(parts);result.write_parquet(dest/'oof.parquet');rows=[]
for method in ['generic','available_family_max']:
    ranked=result.sort(['pair_id',method,'hand_id'],descending=[False,True,False])
    for key,z in ranked.group_by('pair_id',maintain_order=True):
        rel=z['evidence'].to_numpy()[:5];score=float((np.cumsum(rel)/np.arange(1,len(rel)+1)*rel).sum()/min(5,z['evidence'].sum()));rows.append(dict(pair_id=key[0],family=z['behavior_family'][0],method=method,map5=score))
df=pd.DataFrame(rows);df.to_csv(dest/'validation.csv',index=False);print('RESULT',df.groupby(['method','family']).map5.mean().to_string(),flush=True);print('OVERALL',df.groupby('method').map5.mean().to_dict(),flush=True)
