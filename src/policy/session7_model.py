from pathlib import Path
import joblib
import numpy as np,polars as pl
from scipy.special import softmax
from catboost import CatBoostClassifier
from session6_priority_model import load_models as load_old
from session7_likelihood import multilevel_inclusion
ROOT=Path('artifacts/evidence_session7')
def load_models(method):
    if method=='event_fusion':
        em,cols=load_models('em1');hist,_=load_models('hist_eventblend');return {'__fusion__':(em,hist)},cols
    old,cols=load_old('priority_ordered');result={}
    for b in old:
        result[b]=[]
        for f in range(4):
            if method in ['hist_events','hist_eventblend']:
                heads=[joblib.load(ROOT/f'hist_event{k}_{b}_fold{f}.joblib') for k in (1,2)];assert all(m.n_features_in_==len(cols) for m in heads);result[b].append((method,heads,old[b][f]))
            elif method=='calibration':result[b].append(('calibration',old[b][f],np.load(ROOT/f'calibration_{b}_fold{f}.npy')))
            elif method.startswith('three_tier') and b!='soft_play':result[b].append(('old',old[b][f],None))
            else:
                m=CatBoostClassifier();m.load_model(str(ROOT/f'{method}_{b}_fold{f}.cbm'));assert len(m.feature_names_)==len(cols);result[b].append(('categorical',m,None))
    return result,cols
def score(models,cols,b,q,folds):
    if '__fusion__' in models:
        em,hist=models['__fusion__'];return .5*score(em,cols,b,q,folds)+.5*score(hist,cols,b,q,folds)
    assert q.select('pair_id','hand_id').n_unique()==len(q)
    g=q.with_row_index('__original').sort('pair_id','time','hand_id');X=g.select(cols).to_numpy();assert np.isfinite(X).all();groups=[z['__original'].to_numpy() for _,z in g.group_by('pair_id',maintain_order=True)];answer=np.zeros(len(q));folds=list(folds)
    for f in folds:
        mode,model,weights=models[b][f]
        if mode=='categorical':p=model.predict_proba(X,thread_count=4)
        elif mode in ['hist_events','hist_eventblend']:
            a=model[0].predict_proba(X)[:,1];bb=model[1].predict_proba(X)[:,1];scale=np.maximum(1,a+bb);a/=scale;bb/=scale;p=np.stack([np.maximum(0,1-a-bb),a,bb],1)
            if mode=='hist_eventblend':
                ca=weights[0].predict_proba(X,thread_count=4)[:,1];cb=weights[1].predict_proba(X,thread_count=4)[:,1];scale=np.maximum(1,ca+cb);ca/=scale;cb/=scale;p=.5*p+.5*np.stack([np.maximum(0,1-ca-cb),ca,cb],1)
        else:
            a=model[0].predict_proba(X,thread_count=4)[:,1];bb=model[1].predict_proba(X,thread_count=4)[:,1];scale=np.maximum(1 if mode=='old' else 1.000001,a+bb);a/=scale;bb/=scale;none=np.maximum(0,1-a-bb);p=np.stack([none,a,bb],1)
            if mode=='calibration':
                z=(np.log(np.maximum(1e-8,p[:,1:]))-np.log(np.maximum(1e-8,p[:,0]))[:,None])*weights[:2]+weights[2:];p=softmax(np.column_stack([np.zeros(len(q)),z]),axis=1)
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
        off=0
        for ix in groups:answer[ix]+=multilevel_inclusion(p[off:off+len(ix)])/len(folds);off+=len(ix)
    return answer
