import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session25_persistent_actor import actor_features,pair_features,ATTR
ROOT=Path('artifacts/evidence_session25_persistent_actor');C=pl.col
def main():
 full=hand_data();d=full.filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];pids=[d['pair_id'][int(ix[0])] for ix in groups];fold=np.array([d['fold'][int(ix[0])] for ix in groups]);fv=d['fold'].to_numpy();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();z=np.load(ROOT/'actor_features.npz');AX=actor_features(d);PX=pair_features(AX,groups);assert np.array_equal(AX,z['hand']);assert np.array_equal(PX,z['pair']);n=len(ATTR);assert np.array_equal(AX[:,0,:n],AX[:,1,n:]);assert np.array_equal(AX[:,0,n:],AX[:,1,:n]);assert np.array_equal(pair_features(AX[:,::-1],groups),PX[:,::-1]);saved=pl.DataFrame({'pair_id':pids}).join(pl.read_parquet(ROOT/'actor_oof.parquet'),on='pair_id',validate='1:1',maintain_order='left');cond=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'conditional_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.;eventerr=0.;rows=[]
 for f in range(4):
  assignments=json.load(open(ROOT/f'assignments_fold{f}.json'));assert not set(a['pair_id'] for a in assignments)&set(np.array(pids)[fold==f]);vi=np.flatnonzero(fv==f);dm=CatBoostClassifier();dm.load_model(str(ROOT/f'actor_fold{f}.cbm'));p=dm.predict_proba(PX[fold==f].reshape(-1,PX.shape[-1]),thread_count=2)[:,1].reshape(-1,2);p/=p.sum(1)[:,None];want=saved.filter(C('fold')==f).select('actor0','actor1').to_numpy();err=max(err,float(abs(p-want).max()));swap=dm.predict_proba(PX[fold==f,::-1].reshape(-1,PX.shape[-1]),thread_count=2)[:,1].reshape(-1,2);swap/=swap.sum(1)[:,None];assert np.array_equal(p,swap[:,::-1]);dw=np.zeros((len(d),2))
  for i,v in zip(np.flatnonzero(fold==f),p):dw[groups[i]]=v
  for kind in ['control','oriented']:
   out=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
   for k in range(2):
    m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'event{k+1}_directed_transfer_fold{f}.cbm'))
    if kind=='control':pr=m.predict_proba(X[vi],thread_count=2)[:,1]
    else:
     pp=np.column_stack([m.predict_proba(np.column_stack([X[vi],AX[vi,r]]),thread_count=2)[:,1] for r in range(2)]);assert np.array_equal(pp,cond[vi].select(f'actor0_event{k}',f'actor1_event{k}').to_numpy());pr=(pp*dw[vi]).sum(1)
    eventerr=max(eventerr,float(abs(pr-out['bg_primary' if k==0 else 'bg_secondary'].to_numpy()[vi]).max()))
  known=(saved['fold'].to_numpy()==f)&(saved['audit_truth_actor'].to_numpy()>=0);pp=saved.select('actor0','actor1').to_numpy()[known];y=saved['audit_truth_actor'].to_numpy()[known];rows.append({'fold':f,'known_actor_pairs':int(known.sum()),'actor_accuracy':float((pp.argmax(1)==y).mean()),'actor_logloss':float(-np.log(pp[np.arange(len(y)),y].clip(1e-9,1)).mean())})
 assert max(err,eventerr)<1e-12;report={'actor_models_replayed':4,'event_models_replayed':16,'actor_probability_error':err,'event_probability_error':eventerr,'feature_rebuild_error':0.,'role_swap_error':0.,'outer_validation_pairs_in_training_assignments':0,'actor_diagnostics':rows,'limitation':'labels reused for hypothesis discovery; inference does not consume saved audit_truth_actor'};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
