import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,importlib.util
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier
from evidence_data import load

root=Path('artifacts/policy');dest=Path('artifacts/role_model');dest.mkdir(exist_ok=True)
C=pl.col;names=['none','directed_transfer','soft_play','coordinated_isolation']
spec=importlib.util.spec_from_file_location('metric','artifacts/reference_metric/reference_metric.py');metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
d,_=load();d=d.join(pl.read_parquet(list((root/'outcome_roles').glob('T*.parquet'))),on=['pair_id','hand_id']).join(pl.read_parquet(list((root/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'])
cols=json.loads((root/'relationship_evidence/columns.json').read_text());folds=json.loads(Path('artifacts/folds.json').read_text())
p=pd.read_csv(root/'directional/oof.csv');p=p[(p.window=='full')&(p.model=='combined')].sort_values('pair_id');p['family']=p[names[1:]].idxmax(axis=1);p['risk_score']=1-p.none
query=d.join(pl.from_pandas(p[['pair_id','family']]),on='pair_id');parts=[]
for f in folds:
    for family in names[1:]:
        q=query.filter(C('table_id').is_in(f['valid_tables'])&(C('family')==family))
        if not len(q):continue
        m=CatBoostClassifier();m.load_model(str(root/f'relationship_evidence/{family}_fold{f["fold"]}.cbm'))
        parts.append(q.select('pair_id','hand_id').with_columns(pl.Series('score',m.predict_proba(q.select(cols).to_numpy(),thread_count=4)[:,1])))
ranked=pl.concat(parts).sort(['pair_id','score','hand_id'],descending=[False,True,False]);chosen=ranked.group_by('pair_id',maintain_order=True).agg(C('hand_id').head(5));chosen.write_parquet(dest/'selected_development_evidence.parquet');choices=dict(chosen.iter_rows())
truth=pd.read_csv('data/development_labels.csv').rename(columns={'label':'risk_score','behavior_family':'predicted_behavior'})
sub=pd.DataFrame(dict(pair_id=p.pair_id,risk_score=p.risk_score,predicted_behavior=np.where(p.risk_score>=.01,p.family,'none')))
ev=pd.read_csv('data/development_evidence.csv').sort_values('evidence_rank').groupby('pair_id').hand_id.apply(list).to_dict()
for j in range(5):
    c=f'evidence_hand_{j+1}';truth[c]=[ev.get(pid,[])[j] if len(ev.get(pid,[]))>j else 'NO_EVIDENCE' for pid in truth.pair_id];sub[c]=[choices.get(pid,[])[j] if len(choices.get(pid,[]))>j else 'NO_EVIDENCE' for pid in sub.pair_id]
official=metric.score(truth,sub,'pair_id');y=p.truth.to_numpy();r=p.risk_score.to_numpy();pred=sub.predicted_behavior.to_numpy()
pap=metric._average_precision(y>0,r);bm=np.mean([metric._average_precision(y==k,r*(pred==names[k])) for k in [1,2,3]])
emap=(official-.7*pap-.1*bm)/.2
report=dict(pair_AP=pap,behavior_MAP=float(bm),evidence_MAP5=float(emap),official_combined_score=official,v4_combined_score=json.loads((root/'combined_validation.json').read_text())['combined_score'],status='development_model_only_unscored_on_Kaggle',limitation='Public folds reused during exploration. Directional classifier worsens most withheld-family tests. No claim of >0.905 Kaggle score.')
(dest/'validation.json').write_text(json.dumps(report,indent=2));sub.to_csv(dest/'development_oof_submission.csv',index=False);print(report,flush=True)
                                                                                  
pc=json.loads((root/'directional/combined_columns.json').read_text())
a=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(C('phase')=='evaluation').join(pl.scan_parquet(str(root/'directional/full/*.parquet')),on=['pair_id','phase']).collect().sort('pair_id');x=a.select(pc).to_numpy();pred=[]
for f in range(4):
    m=CatBoostClassifier();m.load_model(str(root/f'directional/combined_fold{f}.cbm'));pred.append(m.predict_proba(x,thread_count=4))
probs=np.mean(pred,axis=0);out=pd.DataFrame(probs,columns=names).assign(pair_id=a['pair_id'].to_list());template=pd.read_csv('data/evaluation_pairs.csv');out=template[['pair_id']].merge(out,on='pair_id',validate='one_to_one');assert len(out)==112540 and out.notna().all().all();out.to_csv(dest/'evaluation_pair_predictions.csv',index=False)
                                                                                               
v=pd.read_csv(root/'relationship_evidence/validation.csv');tables=d.select('pair_id','table_id').unique().to_pandas();v=v.merge(tables,on='pair_id');pivot=v.pivot(index=['pair_id','table_id'],columns='weight',values='map5');delta=(pivot[1]-pivot[0]).rename('delta').reset_index();group=delta.groupby('table_id').delta.agg(['sum','count']);rng=np.random.default_rng(10911);ix=rng.integers(len(group),size=(2000,len(group)));samples=group['sum'].to_numpy()[ix].sum(1)/group['count'].to_numpy()[ix].sum(1)
bootstrap=dict(mean_gain=float(delta.delta.mean()),pool_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),caveat='Fixed predictions only; excludes retraining and selection uncertainty.')
(dest/'evidence_bootstrap.json').write_text(json.dumps(bootstrap,indent=2));print('evidence bootstrap',bootstrap,flush=True)
