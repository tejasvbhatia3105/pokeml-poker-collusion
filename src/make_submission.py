import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import polars as pl,numpy as np,pandas as pd,json,time
from catboost import CatBoostClassifier
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--pair-predictions',default='artifacts/pair_baseline_eval.csv')
parser.add_argument('--output',default='submission.csv')
args=parser.parse_args()
NAMES=['directed_transfer','soft_play','coordinated_isolation']
cols=json.loads(Path('artifacts/rank_columns.json').read_text())
models={}
for n in NAMES:
    models[n]=[]
    for path in sorted(Path('artifacts').glob(f'rank_{n}_fold*.cbm')):
        m=CatBoostClassifier();m.load_model(path);models[n].append(m)
    assert models[n],n
p=pl.read_csv(args.pair_predictions).with_columns((1-pl.col('none')).alias('risk_score'))
a=p.select(NAMES).to_numpy().argmax(1)
p=p.with_columns(pl.Series('family',[NAMES[i] for i in a]))
selections=[];t=time.time()
for i,path in enumerate(sorted(Path('artifacts/detail_features').glob('*.parquet'))):
    d=pl.read_parquet(path).filter(pl.col('phase')=='evaluation').join(p.select('pair_id','family'),on='pair_id').with_columns(((pl.col('time')-.6)/.4).alias('relative_time'))
    for n in NAMES:
        z=d.filter(pl.col('family')==n)
        if not len(z):continue
        X=z.select(cols).to_numpy();pr=np.mean([m.predict_proba(X,thread_count=5)[:,1] for m in models[n]],axis=0)
        z=z.select('pair_id','hand_id').with_columns(pl.Series('hand_score',pr)).sort(['pair_id','hand_score','hand_id'],descending=[False,True,False])
        selections.append(z.group_by('pair_id',maintain_order=True).agg(pl.col('hand_id').head(5).alias('hands')))
    if i%40==0:print(i,round(time.time()-t,1),flush=True)
e=pl.concat(selections).with_columns([pl.col('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(f'evidence_hand_{i+1}') for i in range(5)]).drop('hands')
r=p.select('pair_id','risk_score',pl.when(pl.col('risk_score')<.01).then(pl.lit('none')).otherwise(pl.col('family')).alias('predicted_behavior')).join(e,on='pair_id',how='left').fill_null('NO_EVIDENCE')
template=pl.read_csv('data/sample_submission.csv')
r=template.select('pair_id').join(r,on='pair_id',how='left',maintain_order='left').select(template.columns)
r.write_csv(args.output);print('saved',r.shape,flush=True)
