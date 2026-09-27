import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from scipy.special import expit,logit
from session48_candidate_selection import ROOT,physical,design
from session8_data import hand_data
C=pl.col
def main():
 d=hand_data();PX,cols=physical(d);np.testing.assert_array_equal(PX,np.load(ROOT/'physical_features.npz')['x']);assert cols==json.load(open(ROOT/'physical_columns.json'));fv=d['fold'].to_numpy();ref=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');old=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/background_oof.parquet').select('pair_id','hand_id','cat_and_joint'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['cat_and_joint'].to_numpy();err=0;models=0
 for f in range(4):
  X,pr,sel,mins=design(d,f);va=fv==f;sv=va&sel;mut=d.with_columns(pl.when(C('fold')==f).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when(C('fold')==f).then(999).otherwise(C('evidence_rank')).alias('evidence_rank'),pl.when(C('fold')==f).then(0).otherwise(C('subtype')).alias('subtype'));xm,pm,sm,km=design(mut,f);np.testing.assert_array_equal(X,xm);np.testing.assert_array_equal(pr,pm);np.testing.assert_array_equal(sel,sm);assert mins==km;p0=logit(pr.clip(1e-7,1-1e-7));cached=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/f'design_fold{f}.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');np.testing.assert_array_equal(pr,cached['prior'].to_numpy());np.testing.assert_array_equal(sel,cached['selected'].to_numpy())
  for kind in ['compact','paired']:
   xx=X if kind=='compact' else np.column_stack([X,PX]);m=CatBoostClassifier();m.load_model(str(ROOT/f'{kind}_fold{f}.cbm'));delta=m.predict(xx[sv],prediction_type='RawFormulaVal',thread_count=2);p=pr.copy();p[sv]=expit(p0[sv]+delta);t=old.copy();t[sv]=expit(logit(old[sv].clip(1e-7,1-1e-7))+delta);err=max(err,float(abs(p[va]-ref[kind].to_numpy()[va]).max()),float(abs(t[va]-ref[kind+'_transport'].to_numpy()[va]).max()));models+=1
 assert err==0;report={'models_replayed':models,'prediction_error':err,'heldout_label_rank_subtype_mutation_leaves_design_exact':True,'physical_features_reassembled_exact':True,'candidate_cache_replay_exact':True,'limitation':'transport changes the inference prior from the weaker analytic training prior; repeated-label hypothesis selection'};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
