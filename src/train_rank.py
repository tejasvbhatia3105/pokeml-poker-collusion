import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import polars as pl,numpy as np,pandas as pd,json,time,argparse
from catboost import CatBoostClassifier
from train_evidence import NAMES,evidence_map
p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=1);args=p.parse_args()
while len(list(Path('artifacts/detail_features').glob('*.parquet')))<400:time.sleep(3)
cache=Path('artifacts/dev_detail.parquet')
if not cache.exists():pl.scan_parquet('artifacts/detail_features/*.parquet').filter(pl.col('phase')=='development').collect().write_parquet(cache)
d=pl.read_parquet(cache).join(pl.read_csv('data/development_labels.csv').select('pair_id','behavior_family'),on='pair_id').join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(pl.col('evidence').fill_null(0)).sort('pair_id','hand_id')
d=d.with_columns((pl.col('time')/.6).alias('relative_time'))
cols=[c for c in d.columns if c not in ['pair_id','hand_id','phase','table_id','behavior_family','evidence','time','net_direction']]
Path('artifacts/rank_columns.json').write_text(json.dumps(cols))
folds=json.loads(Path('artifacts/folds.json').read_text());truth={}
for pair,h,n in pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id','behavior_family').iter_rows():
    if pair not in truth:truth[pair]=(n,set())
    truth[pair][1].add(h)
records=[];preds=[];importance=[]
for f in folds[:args.folds]:
    for j,behavior in enumerate(NAMES[1:],1):
        data=d.filter(pl.col('behavior_family')==behavior);X=data.select(cols).to_numpy();y=data['evidence'].to_numpy();va=data['table_id'].is_in(f['valid_tables']).to_numpy();tr=~va
        m=CatBoostClassifier(iterations=900,depth=5,learning_rate=.035,loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=8,verbose=False,thread_count=5,random_seed=100+f['fold'])
        model_path=Path(f"artifacts/rank_{behavior}_fold{f['fold']}.cbm")
        if model_path.exists():m.load_model(model_path)
        else:m.fit(X[tr],y[tr],eval_set=(X[va],y[va]),early_stopping_rounds=140)
        p=m.predict_proba(X[va])[:,1];prob=np.zeros((len(p),4));prob[:,j]=p
        rr=evidence_map(data.filter(pl.Series(va)),prob,truth);records+=rr
        print('fold',f['fold'],behavior,'iter',m.best_iteration_,'MAP5',np.mean([r[2] for r in rr]),flush=True)
        m.save_model(f"artifacts/rank_{behavior}_fold{f['fold']}.cbm")
        preds.append(data.filter(pl.Series(va)).select('pair_id','hand_id').with_columns(pl.Series('score',p),pl.lit(behavior).alias('behavior')))
        importance.append(pd.DataFrame({'feature':cols,'importance':m.feature_importances_,'behavior':behavior,'fold':f['fold']}))
print('overall',np.mean([r[2] for r in records]),flush=True)
pd.DataFrame(records,columns=['pair_id','behavior','map5','hands']).to_csv('artifacts/rank_validation.csv',index=False)
pl.concat(preds).write_parquet('artifacts/rank_oof.parquet')
pd.concat(importance).to_csv('artifacts/rank_importance.csv',index=False)
