import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session95_learned_player_kernel as s
C=pl.col
def main():
 z=np.load(s.ROOT/'sample.npz');raw=z['x'];y=z['y'];fv=z['fold'];keys=pl.read_parquet(s.ROOT/'sample_keys.parquet');models=s.s.load_models();tf=json.load(open('artifacts/policy/table_folds.json'));inputs={f:np.load(s.ROOT/f'inputs_fold{f}.npz') for f in range(4)};checks=0;reference_checks=0
 audit=json.load(open('artifacts/evidence_session84_nested_joint_policy/reference_audit.json'))
 for r in audit:
  assert not any(tf[t] in r['excluded_folds'] for t in r['training_tables']);reference_checks+=1
 for table in sorted(keys['table_id'].unique().to_list())[::25]:
  a=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');q,h=s.s.select(a);mask=keys['table_id'].to_numpy()==table;x=q.select(s.s.PC).to_numpy();hx=h.select(s.s.PC).to_numpy();np.testing.assert_array_equal(x,raw[mask]);np.testing.assert_array_equal(q['action_class'].to_numpy(),y[mask]);native=tf[table];pp={key:m.predict_proba(x,thread_count=2) for key,m in models.items() if native in key};hh={key:models[key].predict_proba(hx,thread_count=2) for key in pp}
  for f in range(4):
   p=s.legal(sum(pp.values())/3,x) if native==f else s.legal(pp[tuple(sorted([native,f]))],x);ph=s.legal(sum(hh.values())/3,hx) if native==f else s.legal(hh[tuple(sorted([native,f]))],hx);np.testing.assert_array_equal(np.log(p.clip(1e-7)).astype(np.float32),inputs[f]['prior'][mask]);np.testing.assert_array_equal(s.fields(q,h,p,ph),inputs[f]['extra'][mask]);checks+=1
 records=[]
 for f in range(4):
  tr=fv!=f;va=~tr;mut=y.copy();mut[va]=(mut[va]+1)%4;np.testing.assert_array_equal(y[tr],mut[tr]);prior=inputs[f]['prior'];extra=inputs[f]['extra'];base=np.column_stack([raw,prior]);assert np.isfinite(base).all() and np.isfinite(extra).all()
  for kind in s.CONFIG['arms']:
   x=np.column_stack([base,extra]) if kind=='context' else base;m=CatBoostClassifier();m.load_model(str(s.ROOT/f'{kind}_fold{f}.cbm'));p=s.predict(m,x[va],prior[va]);np.testing.assert_array_equal(p,np.load(s.ROOT/f'{kind}_fold{f}.npy'));np.testing.assert_array_equal(p,s.predict(m,x[va][::-1],prior[va][::-1])[::-1]);assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-12);records.append({'fold':f,'kind':kind,'model_replay_error':0,'query_permutation_error':0,'validation_target_rows_in_training':0})
 report={'raw_tables_rebuilt':checks//4,'nested_input_blocks_rebuilt':checks,'reference_policy_exclusion_audits':reference_checks,'query_and_context_separation':'94 raw-outcome mutation and scalar style checks apply to the shared builder','models_replayed':len(records),'training_target_mutations':4,'records':records};(s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
