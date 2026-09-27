import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time
import numpy as np
import polars as pl
import pandas as pd
from catboost import CatBoostClassifier
from evidence_data import load

root=Path('artifacts/policy');dest=root/'outcome_roles';C=pl.col
d,oldcols=load(); extra=pl.read_parquet(list(dest.glob('T*.parquet')))
if os.environ.get('RELATIONSHIP_CONTEXT'):
    dest=root/'relationship_evidence'
    extra=extra.join(pl.read_parquet(list(dest.glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
assert extra.select('pair_id','hand_id').n_unique()==len(extra)
d=d.join(extra,on=['pair_id','hand_id'],how='left')
nc=[c for c in extra.columns if c.startswith(('outcome_','relationship_'))]
assert d.select(nc).null_count().to_numpy().sum()==0
cols=oldcols+nc; (dest/'columns.json').write_text(json.dumps(cols))
folds=json.loads(Path('artifacts/folds.json').read_text());parts=[];models=[];t=time.time()
for f in folds:
    for behavior in ['directed_transfer','soft_play','coordinated_isolation']:
        q=d.filter(C('behavior_family')==behavior); X=q.select(cols).to_numpy();y=q['evidence'].to_numpy();va=q['table_id'].is_in(f['valid_tables']).to_numpy()
        m=CatBoostClassifier(iterations=850,depth=5,learning_rate=.035,loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=8,
            thread_count=4,random_seed=710+f['fold'],verbose=False,allow_writing_files=False)
        m.fit(X[~va],y[~va],eval_set=(X[va],y[va]),early_stopping_rounds=120)
        m.save_model(str(dest/f'{behavior}_fold{f["fold"]}.cbm'));models.append(m)
        parts.append(q.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence').with_columns(pl.Series('score',m.predict_proba(X[va])[:,1])))
        print('fit evidence',f['fold'],behavior,m.tree_count_,round(time.time()-t,1),flush=True)
out=pl.concat(parts).join(pl.read_parquet(root/'sequence/evidence_oof.parquet').select('pair_id','hand_id',(.25*C('old_score')+.75*C('new_score')).alias('v4')),on=['pair_id','hand_id'])
out.write_parquet(dest/'oof.parquet')
records=[]
for weight in [0,.5,1]:
    ranked=out.with_columns(((1-weight)*C('v4')+weight*C('score')).alias('blend')).sort(['pair_id','blend','hand_id'],descending=[False,True,False])
    for (pid,),q in ranked.group_by('pair_id',maintain_order=True):
        rel=q['evidence'].to_numpy()[:5]; score=float((rel*np.cumsum(rel)/np.arange(1,len(rel)+1)).sum()/min(5,q['evidence'].sum()))
        records.append(dict(weight=weight,pair_id=pid,behavior=q['behavior_family'][0],map5=score))
pd.DataFrame(records).to_csv(dest/'validation.csv',index=False)
pd.DataFrame(dict(feature=cols,importance=np.mean([m.feature_importances_ for m in models],axis=0))).sort_values('importance',ascending=False).to_csv(dest/'importance.csv',index=False)
print(pd.DataFrame(records).groupby(['weight','behavior']).map5.mean().to_string(),flush=True)
print('OVERALL',pd.DataFrame(records).groupby('weight').map5.mean().to_dict(),flush=True)
