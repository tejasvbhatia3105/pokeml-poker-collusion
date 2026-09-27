import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from scipy.special import logit,expit
from session66_current_candidates import ROOT,INPUT,hand_data,design,C
def main():
 d=hand_data();px=np.load('artifacts/evidence_session62_grounded_list_boost/grounded_features.npz')['x'];saved=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');old=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id','equal'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['equal'].to_numpy();fv=d['fold'].to_numpy();error=0;models=0
 for f in range(4):
  x,pr,selected,mins=design(d,f,INPUT);record=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/f'design_fold{f}.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');np.testing.assert_array_equal(pr,record['prior'].to_numpy());np.testing.assert_array_equal(selected,record['selected'].to_numpy());changed=d.with_columns(pl.when(C('fold')==f).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when(C('fold')==f).then(999).otherwise(C('evidence_rank')).alias('evidence_rank'),pl.when(C('fold')==f).then(99).otherwise(C('subtype')).alias('subtype'));xx,pp,ss,mm=design(changed,f,INPUT);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(pr,pp);np.testing.assert_array_equal(selected,ss);assert mins==mm;va=fv==f;sv=va&selected
  for kind in ['compact','grounded']:
   X=x if kind=='compact' else np.column_stack([x,px]);m=CatBoostClassifier();m.load_model(str(ROOT/f'{kind}_fold{f}.cbm'));delta=m.predict(X[sv],prediction_type='RawFormulaVal',thread_count=2);p=pr.copy();p[sv]=expit(logit(pr[sv].clip(1e-7,1-1e-7))+delta);t=old.copy();t[sv]=expit(logit(old[sv].clip(1e-7,1-1e-7))+delta);error=max(error,float(abs(p[va]-saved[kind].to_numpy()[va]).max()),float(abs(t[va]-saved[kind+'_transport'].to_numpy()[va]).max()));models+=1
                                                                        
  _,legacy,ls,_=design(d,f);ref=d.select('pair_id','hand_id').join(pl.read_parquet(f'artifacts/evidence_session48_candidate_selection/design_fold{f}.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');np.testing.assert_array_equal(legacy,ref['prior'].to_numpy());np.testing.assert_array_equal(ls,ref['selected'].to_numpy())
 assert error==0;out={'saved_models_replayed':models,'probability_error':error,'all_current_priors_and_shortlists_replayed_exact':True,'four_heldout_label_rank_subtype_mutation_checks':True,'legacy48_default_prior_and_shortlist_replay_exact_all_folds':True,'no_truth_dependent_positive_promotion':True};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
