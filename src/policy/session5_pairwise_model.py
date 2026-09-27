import json
from pathlib import Path
import numpy as np,polars as pl,joblib
from scipy.special import logit,expit
from catboost import CatBoostClassifier
from session4_set_evidence import design,UNAMES,FAMILIES
ROOT=Path('artifacts/evidence_session5')
def load_models(backend='cat'):
    prefix='pairwise' if backend=='cat' else 'pairwise_'+backend;models={}
    for b in FAMILIES:
        for f in range(4):
            cols=json.loads((ROOT/f'{prefix}_{b}_fold{f}_columns.json').read_text())
            if backend=='hist':m=joblib.load(ROOT/f'{prefix}_{b}_fold{f}.joblib')
            else:
                m=CatBoostClassifier();m.load_model(str(ROOT/f'{prefix}_{b}_fold{f}.cbm'))
            models[b,f]=(m,cols)
    return models
def rankings(g,weights,models,folds,backend='cat'):
    if len(g)<12:
        h=g.sort(['base_score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5]
        return {n:h for n in ['r27','comparator','comparator_quarter','comparator_half']}
    if 'evidence' not in g.columns:g=g.with_columns(pl.lit(0).alias('evidence'))
    if 'fold' not in g.columns:g=g.with_columns(pl.lit(0).alias('fold'))
    b=design(g.with_columns((pl.col('relative_time')*.6).alias('time')));bi=FAMILIES.index(b['family']);n=len(UNAMES)
    prior=b['U']@(weights[:n]+weights[n+bi*n:n+(bi+1)*n]);i,j=np.where(~np.eye(12,dtype=bool));ps=[]
    z=pl.DataFrame({'hand_id':b['hand']}).join(g,on='hand_id',maintain_order='left',validate='1:1')
    assert z['hand_id'].to_list()==b['hand']
    for f in folds:
        m,cols=models[b['family'],f];x=np.concatenate([b['U'],z.select(cols).to_numpy()],1).astype('float32') if cols else b['U']
        xx=np.concatenate([x[i]-x[j],(x[i]+x[j])/2],1)
        if backend=='residual':p=expit(m.predict(xx,prediction_type='RawFormulaVal',thread_count=4)+prior[i]-prior[j])
        elif backend!='hist':p=m.predict_proba(xx,thread_count=4)[:,1]
        else:p=m.predict_proba(xx)[:,1]
        ps.append(p)
    mat=np.zeros((12,12));mat[i,j]=np.mean(ps,axis=0);mat=(mat+1-mat.T)/2
    comparator=logit(np.clip((mat.sum(1)-.5)/11,1e-5,1-1e-5))
    return {name:[b['hand'][k] for k in np.lexsort((np.array(b['hand']),-s))[:5]] for name,s in [('r27',prior),('comparator',comparator),('comparator_quarter',.75*prior+.25*comparator),('comparator_half',.5*prior+.5*comparator)]}
