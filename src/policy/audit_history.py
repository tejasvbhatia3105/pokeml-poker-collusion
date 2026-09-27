import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,polars as pl
from catboost import CatBoostClassifier
root=Path('artifacts/policy');dest=root/'history';dest.mkdir(exist_ok=True);C=pl.col;cols=json.loads((root/'residual_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text());models=[]
for f in range(4):
 m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f}.cbm'));models.append(m)
ev=pl.read_csv(root/'residual_eval.csv').select('pair_id',(1-C('none')).alias('evaluation_risk'));parts=[]
for path in sorted((root/'pair_features').glob('*.parquet')):
 d=pl.read_parquet(path).filter(C('phase')=='development').join(ev.select('pair_id'),on='pair_id')
 if not len(d):continue
 p=models[folds[path.stem]].predict_proba(d.select(cols).to_numpy(),thread_count=3)
 parts.append(d.select('pair_id','policy_n_hands').with_columns(pl.Series('history_risk',1-p[:,0])))
d=ev.join(pl.concat(parts),on='pair_id',how='left');d.write_csv(dest/'evaluation_history.csv');r={'missing_history_pairs':d['history_risk'].null_count(),'evaluation_risk_above_0.9':d.filter(C('evaluation_risk')>.9).height,'both_period_risks_above_0.9':d.filter((C('evaluation_risk')>.9)&(C('history_risk')>.9)).height,'history_above_0.9_current_below_0.1':d.filter((C('history_risk')>.9)&(C('evaluation_risk')<.1)).height,'history_above_0.9_current_below_0.01':d.filter((C('history_risk')>.9)&(C('evaluation_risk')<.01)).height};print(r);(dest/'audit.json').write_text(json.dumps(r,indent=2));print(d.filter(C('history_risk')>.9).sort('evaluation_risk').head(15))
