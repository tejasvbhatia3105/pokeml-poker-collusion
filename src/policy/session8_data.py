import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session7_em import data
from session6_priority import training_targets
from session7_compare import frame
ROOT=Path('artifacts/evidence_session8');C=pl.col
def hand_data():
    return data().join(pl.read_parquet('artifacts/evidence_session6/order_subtype_labels.parquet'),on=['pair_id','hand_id'],how='left',maintain_order='left',validate='1:1').with_columns(C('subtype').fill_null(0))
def targets(d,f,b):
    tc=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['type'];m=CatBoostClassifier();m.load_model(f'artifacts/evidence_session6/priority_ordered_type_{b}_fold{f}.cbm');tp=m.predict_proba(d.select(tc).to_numpy(),thread_count=4)[:,1]
    fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();tr=(fv!=f)&(fam==b);va=(fv==f)&(fam==b);groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)]
    a,bb,ea,eb=training_targets(d['evidence'].to_numpy(),d['subtype'].to_numpy(),tp,tr,d['time'].to_numpy(),groups,d['evidence_rank'].fill_null(0).to_numpy());assert not np.any((ea|eb)&va)
    return a,bb,ea,eb,va
def reference():
    d=frame().join(pl.read_parquet('artifacts/evidence_session7/hist_eventblend_oof.parquet').select('pair_id','hand_id',C('score').alias('histblend')),on=['pair_id','hand_id'],validate='1:1')
    return d.with_columns(((C('r28')+C('histblend'))*.5).alias('r29'))
