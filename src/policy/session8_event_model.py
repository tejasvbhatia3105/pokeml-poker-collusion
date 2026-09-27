import json,joblib
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session6_priority import inclusion
ROOT=Path('artifacts/evidence_session8')
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
def load_models(mode):
    assert mode in ['context','ledger','raw_events','shared_events','no_clock','exposure_cat','exposure_hist']
    cols=json.loads((ROOT/f'{"exposure" if mode.startswith("exposure_") else mode}_columns.json').read_text());models={}
    for b in FAMILIES:
        models[b]=[]
        for f in range(4):
            heads=[]
            for k in (1,2):
                file=ROOT/f'{mode}_event{k}_{"shared" if mode=="shared_events" else b}_fold{f}'
                if mode=='exposure_hist':m=joblib.load(str(file)+'.joblib');assert m.n_features_in_==len(cols)
                else:m=CatBoostClassifier();m.load_model(str(file)+'.cbm');assert len(m.feature_names_)==len(cols)
                heads.append(m)
            models[b].append(heads)
    return models,cols
def score(models,cols,b,q,folds):
    assert q.select('pair_id','hand_id').n_unique()==len(q)
    g=q.with_row_index('__original').sort('pair_id','time','hand_id');X=g.select(cols).to_numpy();assert np.isfinite(X).all()
    groups=[z['__original'].to_numpy() for _,z in g.group_by('pair_id',maintain_order=True)];out=np.zeros(len(q));folds=list(folds)
    for f in folds:
        p=np.column_stack([m.predict_proba(X)[:,1] for m in models[b][f]]);off=0
        for ix in groups:out[ix]+=inclusion(p[off:off+len(ix),0],p[off:off+len(ix),1])/len(folds);off+=len(ix)
    return out
def add_features(d,mode):
    if mode=='shared_events':return d.with_columns(*[(pl.col('behavior_family')==b).cast(pl.Float32).alias('family_'+b) for b in FAMILIES])
    paths={'context':ROOT/'context_hand_features.parquet','ledger':ROOT/'ledger_hand_features.parquet','raw_events':Path('artifacts/evidence_session4/raw_action_features.parquet')}
    if mode in paths:return d.join(pl.read_parquet(paths[mode]),on=['pair_id','hand_id'],validate='m:1',maintain_order='left')
    return d
