import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session78_grounded_types import ROOT,C
from session8_data import hand_data
from session62_grounded_list_boost import grounded
from session55_current_targets import soft_support
def main():
 full=hand_data();gx,cols=grounded(full);mutated=full.with_columns(pl.lit(0).alias('subtype'),pl.lit(0).alias('evidence'),pl.lit(-999).alias('evidence_rank'));mx,mcols=grounded(mutated);assert cols==mcols;np.testing.assert_array_equal(gx,mx);cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));count=0;err=0;checks=[]
 for family in ['directed_transfer','soft_play']:
  mask=full['behavior_family'].to_numpy()==family;d=full.filter(C('behavior_family')==family).drop('row').with_row_index('row');x=gx[mask];xt=d.select(cfg['type']).to_numpy();fv=d['fold'].to_numpy();oof=pl.read_parquet(ROOT/f'{family}_oof.parquet');nc=2 if family=='directed_transfer' else 3
  for f in range(4):
   if nc==3:
    sub,known,_=soft_support(d,fv!=f);np.testing.assert_array_equal(sub[fv!=f],oof['known_type'].to_numpy()[fv!=f]);checks.append({'fold':f,'training_known_labels_match_training_only_construction':True})
   for kind in ['original','grounded','augmented']:
    path=(f'artifacts/evidence_session6/priority_ordered_type_{family}_fold{f}.cbm' if nc==2 else f'artifacts/evidence_session17_grounded/type_fold{f}.cbm') if kind=='original' else str(ROOT/f'{family}_{kind}_fold{f}.cbm');m=CatBoostClassifier();m.load_model(path);X=xt if kind=='original' else x if kind=='grounded' else np.column_stack([xt,x]);p=m.predict_proba(X[fv==f],thread_count=2);want=oof.select([kind+'_'+str(k) for k in range(nc)]).to_numpy()[fv==f];err=max(err,float(abs(p-want).max()));np.testing.assert_array_equal(p,m.predict_proba(X[fv==f][::-1],thread_count=2)[::-1]);count+=1
 assert err==0;proof={'model_replays':count,'probability_error':err,'row_permutation_exact':True,'raw_feature_truth_rank_subtype_invariance':True,'soft_known_target_checks':checks,'direct_label_semantics':'class1 primary; class0 secondary; initial diagnostic inversion corrected before final complete refit'};(ROOT/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
