import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,pandas as pd,polars as pl
from catboost import CatBoostClassifier
from evidence_data import load
root=Path('artifacts/policy');dest=root/'value_evidence';dest.mkdir(exist_ok=True)
d,cols=load();v=pl.scan_parquet(str(root/'value_hands/*.parquet')).filter(pl.col('phase')=='development').join(d.select('pair_id').unique().lazy(),on='pair_id').collect();vc=[c for c in v.columns if c.startswith('value_')]
d=d.join(v.select('pair_id','hand_id',*vc),on=['pair_id','hand_id'],how='left');assert d.select(vc).null_count().to_numpy().sum()==0
cols=cols+vc;(dest/'columns.json').write_text(json.dumps(cols));folds=json.loads(Path('artifacts/folds.json').read_text());parts=[];t=time.time()
for f in folds:
    for family in ['directed_transfer','soft_play','coordinated_isolation']:
        z=d.filter(pl.col('behavior_family')==family);X=z.select(cols).to_numpy();y=z['evidence'].to_numpy();va=z['table_id'].is_in(f['valid_tables']).to_numpy();tr=~va
        m=CatBoostClassifier(iterations=850,depth=5,learning_rate=.035,loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=8,thread_count=4,random_seed=710+f['fold'],verbose=False,allow_writing_files=False)
        m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=120);m.save_model(str(dest/f'{family}_fold{f["fold"]}.cbm'))
        parts.append(z.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence').with_columns(pl.Series('value_score',m.predict_proba(X[va])[:,1])))
        print('fit',f['fold'],family,m.tree_count_,round(time.time()-t,1),flush=True)
result=pl.concat(parts).join(pl.read_parquet(root/'sequence/evidence_oof.parquet').select('pair_id','hand_id',(pl.col('old_score')*.25+pl.col('new_score')*.75).alias('v4_score')),on=['pair_id','hand_id']);result.write_parquet(dest/'oof.parquet');rows=[]
for weight in [0,.25,.5,.75,1]:
    ranked=result.with_columns(((1-weight)*pl.col('v4_score')+weight*pl.col('value_score')).alias('score')).sort(['pair_id','score','hand_id'],descending=[False,True,False])
    for key,g in ranked.group_by('pair_id',maintain_order=True):
        rel=g['evidence'].to_numpy()[:5];score=float((np.cumsum(rel)/np.arange(1,len(rel)+1)*rel).sum()/min(5,g['evidence'].sum()));rows.append(dict(weight=weight,pair_id=key[0],behavior=g['behavior_family'][0],map5=score))
df=pd.DataFrame(rows);df.to_csv(dest/'validation.csv',index=False);print('RESULT',df.groupby('weight').map5.mean().to_dict(),flush=True);print(df.groupby(['weight','behavior']).map5.mean(),flush=True)
