import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session59_pressure_equity import ROOT,COLS,states,equity,fields,canonical,PERMS,C
from session57_isolation_pressure import data,BASE,noisy_or
def main():
 _,d,a,ac=data();cache=np.load(ROOT/'equity_audit.npz');ss,mm,meta=states(d,a);cm=cache['meta'];order=np.argsort(meta[:,0]);cached_order=np.argsort(cm[:,0]);current_indices=meta[order,1:3].ravel();cached_indices=cm[cached_order,1:3].ravel();np.testing.assert_array_equal(ss[current_indices],cache['states'][cached_indices]);np.testing.assert_array_equal(mm[current_indices],cache['masks'][cached_indices]);np.testing.assert_array_equal(meta[order][:,[0,3,4]],cm[cached_order][:,[0,3,4]]);canonical_cases=0
 for cs in ss[np.random.default_rng(5900).choice(len(ss),30,replace=False)]:
  want=canonical(cs)
  for p in PERMS:
   v=[-1 if c<0 else 4*(int(c)//4)+p[int(c)%4] for c in cs];assert canonical(v)==want;canonical_cases+=1
  reversed_holes=sum(([int(cs[2*j+1]),int(cs[2*j])] for j in range(6)),[])+cs[12:].tolist();assert canonical(reversed_holes)==want
 unique=cache['canonical_unique'];selected=np.linspace(0,len(unique)-1,64,dtype=int);replay=equity(unique[selected,:17],unique[selected,17],4096);np.testing.assert_array_equal(replay,cache['precise_equity'][selected]);raw=equity(ss,mm,128);np.testing.assert_array_equal(raw[current_indices],cache['raw_equity'][cached_indices]);ex=fields(cache['precise_equity'][cache['inverse']],cm,a);np.testing.assert_array_equal(ex,np.load(ROOT/'features.npz')['x']);cfg=json.load(open(ROOT/'config.json'));g=a['row'].to_numpy();fv=d['fold'].to_numpy();x=np.column_stack([a.select(ac).to_numpy(),d.select(cfg['hand_columns']).to_numpy()[g],ex]);saved=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');models=0;err=perm=0
 for f in range(4):
  va=fv[g]==f
  for k in range(2):
   for em in range(3):
    m=CatBoostClassifier();m.load_model(str(ROOT/f'event{k+1}_fold{f}_em{em}.cbm'));models+=1;p=m.predict_proba(x[va],thread_count=2)[:,1];hp=noisy_or(p,g[va],len(d));hp2=noisy_or(m.predict_proba(x[va][::-1],thread_count=2)[:,1],g[va][::-1],len(d));perm=max(perm,float(abs(hp-hp2).max()))
    if em==2:err=max(err,float(abs(hp[fv==f]-saved[['bg_primary','bg_secondary'][k]].to_numpy()[fv==f]).max()))
 assert err==0 and perm<1e-12;z=pl.read_parquet(ROOT/'event_oof.parquet').join(pl.read_parquet(BASE/'event_oof.parquet'),on=['pair_id','hand_id'],suffix='_base',validate='1:1').filter(~C('pair_id').is_in(d['pair_id'].unique().implode()))
 for col in ['bg_primary','bg_secondary']:np.testing.assert_array_equal(z[col].to_numpy(),z[col+'_base'].to_numpy())
 prior=json.load(open('artifacts/evidence_session57_isolation_pressure/verification.json'));assert prior['event_probability_error']==0 and prior['pressure_cache_exact_complete_against_raw_actions'];out={'saved_models_replayed':models,'event_probability_error':err,'action_permutation_error':perm,'raw_states_and_masks_exact':True,'all_raw128_equities_replayed_exact':True,'precise4096_state_replays':64,'suit_symmetry_checks':canonical_cases,'hole_order_checks':30,'features_replay_exact':True,'other_families_unchanged':True,'same_verified57_targets_and_support':True};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
