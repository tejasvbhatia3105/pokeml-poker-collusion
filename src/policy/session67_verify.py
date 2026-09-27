import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session67_isolation_types as s
from session57_isolation_pressure import data
from session59_pressure_equity import states,fields,equity
def main():
 _,d,a,ac=data();x,cols=s.design(d,a,ac);np.testing.assert_array_equal(x,np.load(s.ROOT/'features.npz')['x'])
                                                                                
 ss,mm,meta=states(d,a);cached=np.load('artifacts/evidence_session59_pressure_equity/equity_audit.npz');cm=cached['meta'];ix=np.argsort(meta[:,0]);ci=np.argsort(cm[:,0]);np.testing.assert_array_equal(ss[meta[ix,1]],cached['states'][cm[ci,1]]);np.testing.assert_array_equal(mm[meta[ix,1]],cached['masks'][cm[ci,1]]);np.testing.assert_array_equal(meta[ix][:,[0,3,4]],cm[ci][:,[0,3,4]])
                                                                          
 precise=fields(cached['precise_equity'][cached['inverse']],cm,a);np.testing.assert_array_equal(precise,np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x'])
 mutated=d.with_columns(pl.lit(0).alias('subtype'),pl.lit(0).alias('evidence'),pl.lit(0).alias('evidence_rank'),pl.lit(-999).alias('time'));xm,_=s.design(mutated,a,ac);np.testing.assert_array_equal(x,xm)
 xt=d.select(json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['type']).to_numpy();oof=pl.read_parquet(s.ROOT/'type_oof.parquet');fv=d['fold'].to_numpy();error=0;models=0
 for f in range(4):
  va=fv==f
  for kind in ['original','pressure','augmented']:
   m=CatBoostClassifier();path=f'artifacts/evidence_session6/priority_ordered_type_coordinated_isolation_fold{f}.cbm' if kind=='original' else str(s.ROOT/f'{kind}_fold{f}.cbm');m.load_model(path);xx=xt if kind=='original' else x if kind=='pressure' else np.column_stack([xt,x]);p=m.predict_proba(xx[va],thread_count=2)[:,1];error=max(error,float(abs(p-oof[kind].to_numpy()[va]).max()));np.testing.assert_array_equal(p,m.predict_proba(xx[va][::-1],thread_count=2)[:,1][::-1]);models+=1
 assert error==0
 z=d.with_columns(pl.Series('pressure_active_max',x[:,cols.index('players_active_max')])).filter(pl.col('evidence')==1);badcount=badtime=0
 for _,q in z.group_by('pair_id'):
  q=q.sort('evidence_rank');v=q['pressure_active_max'].to_numpy();t=q['time'].to_numpy();badcount+=int((np.diff(v)>0).any());badtime+=int(((np.diff(v)==0)&(np.diff(t)<0)).any())
 report={'model_replays':models,'prediction_error':error,'feature_cache_exact':True,'raw_card_state_alignment_exact':True,'precision_fields_replay_exact':True,'feature_invariance_to_truth_rank_subtype_time':True,'row_permutation_exact':True,'feature_count':len(cols),'active_player_hypothesis':{'provenance':'post-result interpretation of67 feature importance; descriptive over all public truth, not fresh validation','truth_counts':z.group_by('pressure_active_max').len().sort('pressure_active_max').to_dicts(),'pairs_violating_descending_active_count':badcount,'pairs_violating_chronology_within_active_count':badtime}}
 (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
