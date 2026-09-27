import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier,Pool
from evidence_data import load
root=Path('artifacts/policy');dest=root/'query_balanced_evidence';dest.mkdir(exist_ok=True);d,cols=load();C=pl.col
d=d.with_columns(C('evidence').sum().over('pair_id').alias('positive_count'),pl.len().over('pair_id').alias('query_count'))
d=d.with_columns(pl.when(C('evidence')==1).then(.5/C('positive_count')).otherwise(.5/(C('query_count')-C('positive_count'))).alias('weight'))
assert np.isfinite(d['weight'].to_numpy()).all()
folds=json.loads(Path('artifacts/folds.json').read_text());parts=[];t=time.time()
for f in folds:
    for family in ['directed_transfer','soft_play','coordinated_isolation']:
        z=d.filter(C('behavior_family')==family);X=z.select(cols).to_numpy();y=z['evidence'].to_numpy();w=z['weight'].to_numpy();va=z['table_id'].is_in(f['valid_tables']).to_numpy();tr=~va;tw=w[tr]/w[tr].mean();vw=w[va]/w[va].mean()
        m=CatBoostClassifier(iterations=850,depth=5,learning_rate=.035,loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=8,thread_count=4,random_seed=710+f['fold'],verbose=False,allow_writing_files=False)
        m.fit(X[tr],y[tr],sample_weight=tw,eval_set=Pool(X[va],y[va],weight=vw),early_stopping_rounds=120);m.save_model(str(dest/f'{family}_fold{f["fold"]}.cbm'));parts.append(z.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence').with_columns(pl.Series('balanced',m.predict_proba(X[va])[:,1])));print(f['fold'],family,m.tree_count_,round(time.time()-t,1),flush=True)
r=pl.concat(parts).join(pl.read_parquet(root/'sequence/evidence_oof.parquet').select('pair_id','hand_id',(.25*C('old_score')+.75*C('new_score')).alias('v4')),on=['pair_id','hand_id']);r.write_parquet(dest/'oof.parquet');rows=[]
                                                                                      
r=r.with_columns(C('balanced').rank('ordinal',descending=True).over('pair_id').alias('balanced_rank'),C('v4').rank('ordinal',descending=True).over('pair_id').alias('v4_rank'))
for alpha in [0,.25,.5,.75,1]:
    z=r.with_columns((alpha/(C('balanced_rank')+5)+(1-alpha)/(C('v4_rank')+5)).alias('score')).sort(['pair_id','score','hand_id'],descending=[False,True,False])
    for key,g in z.group_by('pair_id',maintain_order=True):
        rel=g['evidence'].to_numpy()[:5];value=float((np.cumsum(rel)/np.arange(1,len(rel)+1)*rel).sum()/min(5,g['evidence'].sum()));rows.append(dict(pair_id=key[0],family=g['behavior_family'][0],balanced_weight=alpha,map5=value))
df=pd.DataFrame(rows);df.to_csv(dest/'validation.csv',index=False);print('RESULT',df.groupby('balanced_weight').map5.mean().to_dict(),flush=True);print(df.groupby(['balanced_weight','family']).map5.mean(),flush=True)
