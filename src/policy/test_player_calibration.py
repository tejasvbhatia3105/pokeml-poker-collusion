\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl
from catboost import CatBoostClassifier
root=Path('artifacts/policy');dest=root/'player_calibration';dest.mkdir(exist_ok=True)
cols=json.loads((root/'feature_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text());models=[]
for f in range(4):
    m=CatBoostClassifier();m.load_model(str(root/f'action_fold{f}.cbm'));models.append(m)
totals={}; per_table=[];t=time.time()
for ti,path in enumerate(sorted((root/'actions').glob('*.parquet'))[::10]):
    a=pl.read_parquet(path).filter(pl.col('phase')=='development');act=a['action_class'].to_numpy();p=models[folds[path.stem]].predict_proba(a.select(cols).to_numpy(),thread_count=4)
    call=a['to_call'].to_numpy()>0;legal=np.ones_like(p);legal[call,1]=0;legal[~call,0]=0;legal[~call,2]=0;p*=legal;p/=p.sum(1,keepdims=True)
    r=np.eye(4)[act]-p
    a=a.with_columns(pl.Series('call',call),((pl.col('equity')*5).floor().clip(0,4)).alias('eq_bin'),*[pl.Series(f'r{k}',r[:,k]) for k in range(4)])
    scores={'baseline':-np.log(p[np.arange(len(a)),act].clip(1e-9)).sum()}
    for scope in ['street','strength','strength_pressure']:
        keys=['player_id','street_no']
        if scope!='street':keys+=['eq_bin']
        if scope=='strength_pressure':keys+=['call']
        z=a.with_columns(*[(pl.col(f'r{k}').sum().over(keys)-pl.col(f'r{k}').sum().over(keys+['hand_id'])).alias(f'b{k}') for k in range(4)],(pl.len().over(keys)-pl.len().over(keys+['hand_id'])).alias('other_n'))
        bias=z.select([f'b{k}' for k in range(4)]).to_numpy();n=z['other_n'].to_numpy()
        for shrink in [10,30,100]:
            q=np.maximum(p+bias/(n[:,None]+shrink),1e-5)*legal;q/=q.sum(1,keepdims=True)
            scores[f'{scope}_{shrink}']=-np.log(q[np.arange(len(a)),act]).sum()
    for k,v in scores.items():totals[k]=totals.get(k,0)+v
    totals['n']=totals.get('n',0)+len(a)
    per_table.append(dict(table_id=path.stem,n=len(a),**{k:v/len(a) for k,v in scores.items()}))
    if ti%10==0:print(ti,'seconds',round(time.time()-t,1),'losses',{k:round(v/totals['n'],6) for k,v in totals.items() if k!='n'},flush=True)
result=dict(tables=len(per_table),actions=totals['n'],logloss={k:v/totals['n'] for k,v in totals.items() if k!='n'},limitations=['Retrospective self-supervised calibration on other hands of each held-out player; not an online forecasting evaluation.','Lower action log loss does not alone establish better coordination detection.'])
(dest/'metrics.json').write_text(json.dumps(result,indent=2));pl.DataFrame(per_table).write_csv(dest/'table_metrics.csv');print('RESULT',result,flush=True)
