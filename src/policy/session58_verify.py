import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session58_pressure_comparison import ROOT,BASE,data,compute,noisy_or,C
def main():
 _,d,a,ac=data();ex,audit=compute(d,a);np.testing.assert_array_equal(ex,np.load(ROOT/'features.npz')['x']);reverse,_=compute(d,a.reverse());np.testing.assert_array_equal(ex,reverse[::-1]);labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');swap=labs.select('pair_id',C('player_2').alias('player_1'),C('player_1').alias('player_2'));swapped,_=compute(d,a,swap);np.testing.assert_array_equal(ex,swapped);cfg=json.load(open(ROOT/'config.json'));g=a['row'].to_numpy();fv=d['fold'].to_numpy();x=np.column_stack([a.select(ac).to_numpy(),d.select(cfg['hand_columns']).to_numpy()[g],ex]);saved=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');models=0;err=perm=0
 for f in range(4):
  va=fv[g]==f
  for k in range(2):
   for em in range(3):
    m=CatBoostClassifier();m.load_model(str(ROOT/f'event{k+1}_fold{f}_em{em}.cbm'));models+=1;p=m.predict_proba(x[va],thread_count=2)[:,1];hp=noisy_or(p,g[va],len(d));hp2=noisy_or(m.predict_proba(x[va][::-1],thread_count=2)[:,1],g[va][::-1],len(d));perm=max(perm,float(abs(hp-hp2).max()))
    if em==2:err=max(err,float(abs(hp[fv==f]-saved[['bg_primary','bg_secondary'][k]].to_numpy()[fv==f]).max()))
 assert err==0 and perm<1e-12
 z=pl.read_parquet(ROOT/'event_oof.parquet').join(pl.read_parquet(BASE/'event_oof.parquet'),on=['pair_id','hand_id'],suffix='_base',validate='1:1').filter(~C('pair_id').is_in(d['pair_id'].unique()))
 for col in ['bg_primary','bg_secondary']:np.testing.assert_array_equal(z[col].to_numpy(),z[col+'_base'].to_numpy())
 prior=json.load(open('artifacts/evidence_session57_isolation_pressure/verification.json'));assert prior['event_probability_error']==0 and prior['pressure_cache_exact_complete_against_raw_actions'];out={'saved_models_replayed':models,'event_probability_error':err,'action_permutation_error':perm,'raw_features_exact':True,'action_reverse_and_endpoint_swap_exact':True,'other_families_unchanged':True,'same_support_and_censored_targets_as_verified57':True,**audit};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
