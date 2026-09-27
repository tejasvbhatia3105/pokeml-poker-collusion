import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session60_aligned_multiway import ROOT,design,MATCHUP,OLD,C
from session59_pressure_equity import states,equity,fields
def main():
 errors=[];models=0;base=pl.read_parquet(MATCHUP/'current/event_oof.parquet');new=pl.read_parquet(ROOT/'event_oof.parquet');prior=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');allpairs=[]
 for family in ['directed_transfer','soft_play']:
  folder=ROOT/family;d,a,x=design(family);cache=np.load(folder/'equity_audit.npz');ss,mm,meta=states(d,a);cm=cache['meta'];order=np.argsort(meta[:,0]);co=np.argsort(cm[:,0]);ix=meta[order,1:3].ravel();ci=cm[co,1:3].ravel();np.testing.assert_array_equal(ss[ix],cache['states'][ci]);np.testing.assert_array_equal(mm[ix],cache['masks'][ci]);np.testing.assert_array_equal(meta[order][:,[0,3,4]],cm[co][:,[0,3,4]]);np.testing.assert_array_equal(equity(ss,mm,128)[ix],cache['raw_equity'][ci]);u=cache['canonical_unique'];sample=np.linspace(0,len(u)-1,64,dtype=int);np.testing.assert_array_equal(equity(u[sample,:17],u[sample,17],4096),cache['precise_equity'][sample]);ex=fields(cache['precise_equity'][cache['inverse']],cm,a);np.testing.assert_array_equal(ex,np.load(folder/'features.npz')['x']);xx=np.column_stack([x,ex]);g=a['row'].to_numpy();r=a['actor'].to_numpy();fv=d['fold'].to_numpy();oldpp=np.zeros((len(d),2));pp=np.zeros_like(oldpp);saved=d.select('pair_id','hand_id').join(pl.read_parquet(folder/'conditional_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0
  for f in range(4):
   va=fv[g]==f;m=CatBoostClassifier();m.load_model(str(folder/f'primary_fold{f}.cbm'));p=m.predict_proba(xx[va],thread_count=2)[:,1];pp[g[va],r[va]]=p;models+=1;control=CatBoostClassifier();control.load_model(str(MATCHUP/'current'/family/f'event1_fold{f}.cbm'));oldpp[g[va],r[va]]=control.predict_proba(x[va],thread_count=2)[:,1];models+=1
  np.testing.assert_array_equal(pp,saved.select('actor0_primary','actor1_primary').to_numpy());dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if family=='directed_transfer' else np.ones_like(pp)
  for frame,v in [(base,oldpp),(new,pp)]:
   z=d.select('pair_id','hand_id').join(frame,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');np.testing.assert_array_equal((v*dw).sum(1),z['bg_primary'].to_numpy())
  allpairs.extend(d['pair_id'].unique().to_list());errors.append({'family':family,'actions':len(a),'raw128_state_replay_exact':True,'precise4096_state_replays':64,'features_replay_exact':True,'R32_matched_control_error':0,'new_primary_replay_error':0})
 z=new.join(prior,on=['pair_id','hand_id'],suffix='_prior',validate='1:1');np.testing.assert_array_equal(z['bg_secondary'].to_numpy(),z['bg_secondary_prior'].to_numpy());q=z.filter(~C('pair_id').is_in(allpairs));np.testing.assert_array_equal(q['bg_primary'].to_numpy(),q['bg_primary_prior'].to_numpy());out={'saved_models_replayed':models,'all_model_and_control_errors':0,'isolation_and_all_secondary_heads_unchanged':True,'families':errors};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
