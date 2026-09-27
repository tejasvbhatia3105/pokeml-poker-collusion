\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,hashlib,time
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session119_private_partner_data as actions
import session122_family_blind_hands as feature_source
from session8_count_conditioning import conditioned

ROOT=Path('artifacts/evidence_session148_generic_inference');C=pl.col

def design(table,query,phase):
    assert phase in ['development','evaluation']
    q=query.select('pair_id','hand_id','player_1','player_2').sort('pair_id','hand_id')
    assert q.select('pair_id','hand_id').n_unique()==len(q) and len(q)>0
    times=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet',columns=['hand_id','time_index','phase']).unique()
    phase_times=times.filter(C('phase')==phase);start=int(phase_times['time_index'].min());length=int(phase_times['time_index'].max())-start+1
    assert phase_times['hand_id'].n_unique()==length
    q=q.join(phase_times.select('hand_id','time_index'),on='hand_id',validate='m:1',maintain_order='left')
    assert q['time_index'].null_count()==0
    q=q.with_row_index('row').with_columns(pl.lit(table).alias('table_id'),pl.lit(-1).alias('fold'),pl.lit(-1).alias('label'),pl.lit('unknown').alias('behavior_family'))
    meta,raw,audit=actions.build(table,q);bag=meta['hand_row'].to_numpy();cls=meta['action_class'].to_numpy();post=raw[:,actions.PUBLIC.index('street_no')]>0
    masks=[np.ones(len(raw),bool),cls==0,(cls==0)&post,cls==2,(cls==2)&post,cls==3,(cls==3)&post]
    pieces=[];cols=[];rawcols=actions.PUBLIC+actions.PRIVATE
    for name,mask in zip(feature_source.GROUPS,masks):
        block,_=feature_source.aggregate(raw,bag,mask,len(q));pieces.append(block)
        cols+=[name+'_log_count']+[name+'_'+stat+'_'+c for stat in ['mean','min','max'] for c in rawcols]
    chronology=np.zeros((len(q),3),np.float32)
    for _,g in q.group_by('pair_id'):
        g=g.sort('time_index','hand_id');ix=g['row'].to_numpy();chronology[ix]=np.column_stack([(g['time_index'].to_numpy()-start)/length,np.arange(len(g))/max(1,len(g)-1),np.full(len(g),np.log1p(len(g)))])
    pieces.append(chronology);cols+=['phase_position','pair_hand_position','log_pair_hand_count'];x=np.column_stack(pieces)
    expected=json.load(open(feature_source.ROOT/'config.json'))['columns'];assert cols==expected and np.isfinite(x).all()
    audit.update(phase=phase,phase_start=start,phase_length=length,features=len(cols),queries=len(q),unknown_metadata_only=True)
    return q.select('pair_id','hand_id','time_index'),x,audit

def load_models(root):
    root=Path(root);models=[]
    for fold in range(4):
        path=root/'all'/f'fold{fold}_em2.cbm';m=CatBoostClassifier();m.load_model(str(path));meta=json.load(open(path.with_suffix('.json')))
        assert np.array_equal(m.classes_,np.arange(3)) and meta['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        models.append((m,meta['minimum'],str(path)))
    return models

def score(q,x,models,folds=range(4)):
    out=q.with_row_index('row');scores=np.zeros(len(q));folds=list(folds);assert folds
    groups=[g.sort('time_index','hand_id')['row'].to_numpy() for _,g in out.group_by('pair_id')]
    for f in folds:
        m,minimum,_=models[f];p=m.predict_proba(x,thread_count=2)
        for ix in groups:scores[ix]+=conditioned(p[ix,1:],minimum)/len(folds)
    assert np.isfinite(scores).all() and scores.min()>=0 and scores.max()<=1+1e-10
    return q.with_columns(pl.Series('score',scores))

def verify():
    ROOT.mkdir(exist_ok=True);d=pl.read_parquet(feature_source.ROOT/'hands.parquet');xold=np.load(feature_source.ROOT/'x.npy',mmap_mode='r');models=load_models('artifacts/evidence_session123_family_holdout')
    tables=[sorted(set(d.filter(C('fold')==f)['table_id']))[0] for f in range(4)];players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');checks=[];start=time.time()
    for table in tables:
        old=d.filter(C('table_id')==table);f=int(old['fold'][0]);query=old.select('pair_id','hand_id').join(players,on='pair_id',validate='m:1');q,x,info=design(table,query,'development')
        z=q.with_row_index('new_row').join(old.select('pair_id','hand_id','hand_index'),on=['pair_id','hand_id'],validate='1:1');assert len(z)==len(old)
        actual=x[z['new_row'].to_numpy()];expected=xold[z['hand_index'].to_numpy()];err=float(abs(actual-expected).max());assert err==0,(table,err)
        pred=score(q,x,models,[f]);reference=pl.read_parquet(f'artifacts/evidence_session123_family_holdout/all/fold{f}.parquet').select('pair_id','hand_id',C('score').alias('expected'))
        j=pred.join(reference,on=['pair_id','hand_id'],validate='1:1');pe=float((j['score']-j['expected']).abs().max());assert pe==0
        reversed_pred=score(q.reverse(),x[::-1],models,[f]);j=pred.join(reversed_pred,on=['pair_id','hand_id'],validate='1:1',suffix='_reversed');re=float((j['score']-j['score_reversed']).abs().max());assert re==0
        swapped=query.select('pair_id','hand_id',C('player_2').alias('player_1'),C('player_1').alias('player_2'))
        sq,sx,_=design(table,swapped,'development');assert q.equals(sq)
        role_error=float(abs(x-sx).max());assert role_error==0
        sp=score(sq,sx,models,[f]);role_prediction_error=float((pred['score']-sp['score']).abs().max());assert role_prediction_error==0
        checks.append(dict(**info,fold=f,raw_feature_error=err,prediction_error=pe,query_reversal_error=re,
            player_role_feature_error=role_error,player_role_prediction_error=role_prediction_error));print('GENERIC_INFERENCE_REPLAY',table,len(q),round(time.time()-start,1),flush=True)
    (ROOT/'verification.json').write_text(json.dumps(dict(method=__doc__,checks=checks,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Four development pools; raw original123 all-family predictor replay. Evaluation normalization uses full evaluation phase extent. No candidate built or quality improvement claimed.'),indent=2));print('COMPLETE',len(checks),flush=True)

if __name__=='__main__':verify()
