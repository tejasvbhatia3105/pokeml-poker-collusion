import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session79_uncertain_targets as s
from session55_current_targets import direct,soft
C=pl.col
def main():
 states=s.state();full=s.hand_data();gx,_=s.grounded(full);count=0;errors=[];checks=[]
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d=v['d'];a=v['a'];g=a['row'].to_numpy();ar=a['actor'].to_numpy();fv=v['fv'];xt=np.column_stack([v['xt'],gx[full['behavior_family'].to_numpy()==fam]]);isdir=fam=='directed_transfer';ax=s.actor_features(d) if isdir else None;dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if isdir else np.ones((len(d),2))
  for f in range(4):
   typ=CatBoostClassifier();typ.load_model(f'artifacts/evidence_session78_grounded_types/{fam}_augmented_fold{f}.cbm');tp=typ.predict_proba(xt,thread_count=2)
   for kind in ['hard','marginal']:
    hard=kind=='hard';want=s.targets(d,fv!=f,tp,fam,hard=hard);mutated=d.with_columns(*[pl.when(C('fold')==f).then(pl.lit(value)).otherwise(C(name)).alias(name) for name,value in [('evidence',0),('evidence_rank',-999),('time',-999),('subtype',0)]]);got=s.targets(mutated,fv!=f,tp,fam,hard=hard)
    for x,y in zip(want[:3],got[:3]):np.testing.assert_array_equal(x,y)
    if hard:
     old=direct(d,fv!=f,tp[:,1]) if isdir else soft(d,fv!=f,tp)
     for i in range(2):np.testing.assert_array_equal(want[i],old[i] if isdir else old[i][:,None])
    y,e,donor,_=want;assert np.all(y<=e+1e-12) and np.all(e<=1+1e-12);saved=d.select('pair_id','hand_id').join(pl.read_parquet(s.ROOT/kind/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    for head in range(2 if isdir else 1):
     m=CatBoostClassifier();m.load_model(str(s.ROOT/kind/f'{fam}_head{head+1}_fold{f}.cbm'));va=fv==f
     if head==0:
      av=fv[g]==f;raw=m.predict_proba(v['x'][av],thread_count=2)[:,1];pr=np.zeros((len(d),2));pr[g[av],ar[av]]=raw;p=(pr*dw).sum(1)[va]
     else:p=(np.column_stack([m.predict_proba(np.column_stack([v['hx'][va],ax[va,r]]),thread_count=2)[:,1] for r in [0,1]])*dw[va]).sum(1)
     errors.append(float(abs(p-saved[['bg_primary','bg_secondary'][head]].to_numpy()[va]).max()));count+=1
    checks.append({'family':fam,'fold':f,'arm':kind,'heldout_mutation_exact':True,'valid_expected_targets':True})
 for kind in ['hard','marginal']:
  z=pl.read_parquet(s.ROOT/kind/'event_oof.parquet').join(pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet'),on=['pair_id','hand_id'],suffix='_base',validate='1:1').join(full.select('pair_id','hand_id',C('behavior_family').alias('verify_family')),on=['pair_id','hand_id'],validate='1:1')
  for col in ['bg_primary','bg_secondary']:
   q=z.filter(C('verify_family')=='coordinated_isolation');np.testing.assert_array_equal(q[col].to_numpy(),q[col+'_base'].to_numpy())
  q=z.filter(C('verify_family')=='soft_play');np.testing.assert_array_equal(q['bg_secondary'].to_numpy(),q['bg_secondary_base'].to_numpy())
 assert max(errors)==0;out={'models_replayed':count,'event_score_error':max(errors),'target_checks':checks,'hard_targets_match78_training_construction':True,'isolation_and_soft_secondary_unchanged':True};(s.ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
