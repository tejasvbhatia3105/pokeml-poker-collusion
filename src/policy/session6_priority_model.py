\
\
\
\
\
\
import json
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from session6_priority import inclusion
from session4_evidence_model import FAMILIES
ROOT=Path('artifacts/evidence_session6')
def load_models(prefix='priority'):
    cols=json.loads((ROOT/f'{prefix}_columns.json').read_text())['event']
    assert not set(cols)&{'evidence','evidence_rank','subtype','pair_id','hand_id','fold'}
    models={}
    for b in FAMILIES:
        models[b]=[]
        for f in range(4):
            heads=[]
            for k in (1,2):
                m=CatBoostClassifier();m.load_model(str(ROOT/f'{prefix}_event{k}_{b}_fold{f}.cbm'))
                assert len(m.feature_names_)==len(cols)
                heads.append(m)
            models[b].append(heads)
    return models,cols
def score(models,cols,b,q,folds=range(4)):
    \
    assert q.select('pair_id','hand_id').n_unique()==len(q)
    assert len(q['behavior_family'].unique())==1 and q['behavior_family'][0]==b
    g=q.with_row_index('__priority_row').sort('pair_id','time','hand_id')
    x=g.select(cols).to_numpy();assert np.isfinite(x).all()
    groups=[z['__priority_row'].to_numpy() for _,z in g.group_by('pair_id',maintain_order=True)]
    sizes=[len(ix) for ix in groups];out=np.zeros(len(q))
    fs=list(folds);assert fs
    for f in fs:
        a=models[b][f][0].predict_proba(x,thread_count=4)[:,1]
        c=models[b][f][1].predict_proba(x,thread_count=4)[:,1]
        off=0
        for original,size in zip(groups,sizes):
            out[original]+=inclusion(a[off:off+size],c[off:off+size])/len(fs);off+=size
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1+1e-12
    return out
