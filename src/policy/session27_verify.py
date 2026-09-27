import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session26_exact_fold_witness import actions
from session27_donor_call_witness import noisy_or
C=pl.col;ROOT=Path('artifacts/evidence_session27_donor_calls');OLD=Path('artifacts/evidence_session25_persistent_actor')
def main():
 d=hand_data().filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');a,ac=actions(d,2);saved=pl.read_parquet(ROOT/'call_actions.parquet');keys=['pair_id','hand_id','actor','action_no'];assert a.sort(keys).equals(saved.sort(keys));a=saved;cfg=json.load(open(ROOT/'columns.json'));assert ac==cfg['action'];g=a['row'].to_numpy();r=a['actor'].to_numpy();fv=d['fold'].to_numpy();hx=d.select(cfg['hand']).to_numpy();AX=np.load(OLD/'actor_features.npz')['hand'];x=np.column_stack([a.select(ac).to_numpy(),hx[g]]);bag=2*g+r;sup=np.bincount(bag,minlength=2*len(d)).reshape(-1,2)>0;dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();errors={}
 for kind in ['call_hand','call_mil']:
  pp=np.zeros((len(d),2))
  for f in range(4):
   m=CatBoostClassifier();m.load_model(str(ROOT/kind/(f'secondary_fold{f}.cbm' if kind=='call_hand' else f'secondary_fold{f}_em2.cbm')));vi=np.flatnonzero(fv==f)
   if kind=='call_hand':
    for j in range(2):
     jj=vi[sup[vi,j]];pp[jj,j]=m.predict_proba(np.column_stack([hx[jj],AX[jj,j]]),thread_count=2)[:,1]
   else:
    va=fv[g]==f;p=m.predict_proba(x[va],thread_count=2)[:,1];pr=noisy_or(p,bag[va],2*len(d)).reshape(-1,2);pp[vi]=pr[vi]
  q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'conditional_secondary.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(pp-q.select('actor0_secondary','actor1_secondary').to_numpy()).max());out=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet').select('pair_id','hand_id','bg_secondary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');weighted=float(abs((pp*dw).sum(1)-out['bg_secondary'].to_numpy()).max());assert max(err,weighted)<1e-12;errors[kind]={'final_models_replayed':4,'conditional_error':err,'actor_weighted_error':weighted}
                                                                             
                                                                                
 p=np.array([.1,.3,.6]);den=1-np.prod(1-p);assert abs(noisy_or(p,np.zeros(3,int),1)[0]-den)<1e-12
 import itertools
 marg=np.zeros(3)
 for v in itertools.product([0,1],repeat=3):
  v=np.array(v);w=np.prod(np.where(v,p,1-p))
  if v.any():marg+=w*v/den
 assert np.max(abs(marg-p/den))<1e-12
 report={'action_rebuild_exact':True,'final_model_replays':errors,'noisy_or_posterior_enumeration':True,'target_audit':json.load(open(ROOT/'audit.json'))};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
