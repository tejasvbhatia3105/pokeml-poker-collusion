import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl,pandas as pd,json
from pathlib import Path
from catboost import CatBoostClassifier
from train_baseline import ap
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--pair-predictions',default='artifacts/pair_oof.csv');parser.add_argument('--report',default='artifacts/combined_metrics.json');args=parser.parse_args()
names=['none','directed_transfer','soft_play','coordinated_isolation']
pair=pd.read_csv(args.pair_predictions).sort_values('pair_id');pair['family']=pair[names[1:]].idxmax(axis=1);pair['risk_score']=1-pair['none']
d=pl.read_parquet('artifacts/dev_detail.parquet').join(pl.from_pandas(pair[['pair_id','family','truth']]),on='pair_id').filter(pl.col('truth')>0).with_columns((pl.col('time')/.6).alias('relative_time'))
cols=json.loads(Path('artifacts/rank_columns.json').read_text());folds=json.loads(Path('artifacts/folds.json').read_text());preds=[]
for f in folds:
    for n in names[1:]:
        z=d.filter(pl.col('table_id').is_in(f['valid_tables'])&(pl.col('family')==n))
        if not len(z):continue
        m=CatBoostClassifier();m.load_model(f"artifacts/rank_{n}_fold{f['fold']}.cbm")
        pr=m.predict_proba(z.select(cols).to_numpy(),thread_count=4)[:,1]
        preds.append(z.select('pair_id','hand_id').with_columns(pl.Series('score',pr)))
r=pl.concat(preds).sort('pair_id','score',descending=[False,True]).group_by('pair_id',maintain_order=True).agg(pl.col('hand_id').head(5))
truth=pl.read_csv('data/development_evidence.csv').group_by('pair_id').agg(pl.col('hand_id')).to_dict(as_series=False);truth=dict(zip(truth['pair_id'],map(set,truth['hand_id'])))
es={}
for pid,hands in r.iter_rows():
    relevant=truth[pid];hits=0;val=0
    for rank,h in enumerate(hands,1):
        if h in relevant:hits+=1;val+=hits/rank
    es[pid]=val/min(5,len(relevant))
risk=pair.risk_score.to_numpy();y=pair.truth.to_numpy();pred=pair.family.to_numpy();emap=np.mean(list(es.values()));bmap=np.mean([ap(y==i,risk*(pred==n)*(risk>=.01)) for i,n in enumerate(names[1:],1)]);pap=ap(y>0,risk)
report={'pair_ap':pap,'evidence_map5':emap,'behavior_map':bmap,'combined_score':.7*pap+.2*emap+.1*bmap,'validation':'Four-fold disjoint player-pool development validation; evidence models select iterations on the validation fold; no other_coordination public positives; not a private test score.'}
Path(args.report).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
