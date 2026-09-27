import json
import numpy as np,polars as pl
from catboost import CatBoostRanker
from scipy.special import expit,logit
from session49_candidate_ranking import ROOT,training_pool
from session48_candidate_selection import physical,design
from session8_data import hand_data
def main():
 d=hand_data();PX,_=physical(d);fv=d['fold'].to_numpy();y=d['evidence'].to_numpy();ref=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');old=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/background_oof.parquet').select('pair_id','hand_id','cat_and_joint'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['cat_and_joint'].to_numpy();err=0;models=0;groups=0
 for f in range(4):
  X,pr,sel,mins=design(d,f);va=fv==f;sv=va&sel;tr=~va&sel;assert not(tr&va).any();gids=d['pair_id'].to_numpy()[tr];assert len(np.unique(gids))==1+np.count_nonzero(gids[1:]!=gids[:-1]);groups+=len(np.unique(gids));mut=d.with_columns(pl.when(pl.col('fold')==f).then(1-pl.col('evidence')).otherwise(pl.col('evidence')).alias('evidence'));xm,pm,sm,km=design(mut,f);np.testing.assert_array_equal(X,xm);np.testing.assert_array_equal(pr,pm);np.testing.assert_array_equal(sel,sm);assert km==mins
  for kind in ['compact','paired']:
   xx=X if kind=='compact' else np.column_stack([X,PX]);m=CatBoostRanker();m.load_model(str(ROOT/f'{kind}_fold{f}.cbm'));delta=m.predict(xx[sv],thread_count=2);p=pr.copy();p[sv]=expit(logit(pr[sv].clip(1e-7,1-1e-7))+delta);t=old.copy();t[sv]=expit(logit(old[sv].clip(1e-7,1-1e-7))+delta);err=max(err,float(abs(p[va]-ref[kind].to_numpy()[va]).max()),float(abs(t[va]-ref[kind+'_transport'].to_numpy()[va]).max()));models+=1
 assert err==0;report={'models_replayed':models,'prediction_error':err,'training_groups_across_folds':groups,'groups_contiguous':True,'heldout_labels_do_not_change_design':True,'validation_training_overlap':0,'limitation':'R31 transport prior shift; repeated-label model selection; PairLogit is a ranking surrogate, not exact MAP optimization'};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
