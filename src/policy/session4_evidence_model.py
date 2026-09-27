from pathlib import Path
import json,joblib
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session4_set_evidence import design,UNAMES,FAMILIES
ROOT=Path('artifacts/evidence_session4')
COLS=json.loads(Path('artifacts/policy/relationship_evidence/columns.json').read_text())
def load_models():
    models={}
    for b in FAMILIES:
        models[b]=[]
        for f in range(4):
            m=CatBoostClassifier();m.load_model(str(ROOT/f'base_{b}_fold{f}.cbm'))
            h=joblib.load(ROOT/f'leafwise_separate_{FAMILIES.index(b)}_fold{f}.joblib')
            models[b].append((m,h))
    return models
def score(models,b,x,folds):
    return np.mean([.5*models[b][f][0].predict_proba(x,thread_count=4)[:,1]+.5*models[b][f][1].predict_proba(x)[:,1] for f in folds],axis=0)
def correct(q,weights):
    if len(q)<12:return q.sort(['base_score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5]
                                                                                
                                                                                 
    g=q.with_columns((pl.col('relative_time')*.6).alias('time'))
    if 'evidence' not in g.columns:g=g.with_columns(pl.lit(0).alias('evidence'))
    if 'fold' not in g.columns:g=g.with_columns(pl.lit(0).alias('fold'))
    b=design(g);f=FAMILIES.index(b['family']);n=len(UNAMES)
    coef=weights[:n]+weights[n+f*n:n+(f+1)*n]
    scores=b['U']@coef;order=np.lexsort((np.array(b['hand']),-scores))[:5]
    return [b['hand'][i] for i in order]
