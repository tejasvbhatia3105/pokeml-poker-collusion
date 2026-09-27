import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session82_seed_stability as s
import session83_joint_policy as j
from session37_bet_fold import paired_features
C=pl.col
def model(path,x):
 m=CatBoostClassifier();m.load_model(str(path));p=m.predict_proba(x,thread_count=2)[:,1];np.testing.assert_array_equal(p,m.predict_proba(x[::-1],thread_count=2)[:,1][::-1]);return p
def main():
 states=s.state();full=s.hand_data();count=0
 for offset in s.OFFSETS:
  saved=pl.read_parquet(s.ROOT/f'seed{offset}'/'event_oof.parquet')
  for fam in ['directed_transfer','soft_play']:
   v=states[fam];d,a=v['d'],v['a'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];direct=fam=='directed_transfer';ax=s.actor_features(d) if direct else None;dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if direct else np.ones((len(d),2));pred=np.zeros((len(d),2))
   for f in range(4):
    va=fv[g]==f;hv=fv==f
    for head in [0,1]:
     path=s.ROOT/f'seed{offset}'/f'{fam}_head{head+1}_fold{f}.cbm'
     if offset==0:path=f'artifacts/evidence_session50_matchup/current/{fam}/event1_fold{f}.cbm' if head==0 else (f'artifacts/evidence_session25_persistent_actor/oriented/event2_directed_transfer_fold{f}.cbm' if direct else f'artifacts/evidence_session6/priority_ordered_event2_soft_play_fold{f}.cbm')
     if head==0:
      p=np.zeros((len(d),2));p[g[va],actor[va]]=model(path,v['x'][va]);pred[hv,0]=(p*dw).sum(1)[hv]
     elif direct:pred[hv,1]=(np.column_stack([model(path,np.column_stack([v['hx'][hv],ax[hv,r]])) for r in [0,1]])*dw[hv]).sum(1)
     else:pred[hv,1]=model(path,v['hx'][hv])
     count+=1
   expect=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left').select('bg_primary','bg_secondary').to_numpy();np.testing.assert_array_equal(pred,expect)
  _,d,a,ac=s.pressure_data();g=a['row'].to_numpy();fv=d['fold'].to_numpy();cfg=json.load(open('artifacts/evidence_session59_pressure_equity/config.json'));x=np.column_stack([a.select(ac).to_numpy(),d.select(cfg['hand_columns']).to_numpy()[g],np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x']]);pred=np.zeros((len(d),2))
  for f in range(4):
   va=fv[g]==f
   for head in [0,1]:
    path=s.ROOT/f'seed{offset}'/f'coordinated_isolation_head{head+1}_fold{f}_em2.cbm' if offset else f'artifacts/evidence_session59_pressure_equity/event{head+1}_fold{f}_em2.cbm';hp=s.noisy_or(model(path,x[va]),g[va],len(d));pred[fv==f,head]=hp[fv==f];count+=1
    if offset:
     for em in [0,1]:
      p=model(s.ROOT/f'seed{offset}'/f'coordinated_isolation_head{head+1}_fold{f}_em{em}.cbm',x[va]);assert np.isfinite(p).all();count+=1
  expect=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left').select('bg_primary','bg_secondary').to_numpy();np.testing.assert_array_equal(pred,expect)
 proof={'models_replayed_including_original_controls_and_intermediate_EM':count,'final_heads_score_error':0,'row_permutation_error':0,'original_seed_event_control_exact':True,'fixed_offsets':[0,10000,20000]};(s.ROOT/'verification.json').write_text(json.dumps(proof,indent=2));print(proof,flush=True)
 saved=pl.read_parquet(j.ROOT/'event_oof.parquet');base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');count=0;action_count=0
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];own,bet,size,pc=j.design(v);cached=np.load(j.ROOT/f'{fam}_raw.npz')
  for n,z in [('own',own),('bet',bet),('size',size)]:np.testing.assert_array_equal(z,cached[n])
  raw,_=paired_features(d,a);np.testing.assert_array_equal(bet,raw.select(['bet_'+c for c in pc]).to_numpy());np.testing.assert_array_equal(size,raw['bet_log_bet_ratio'].to_numpy());assert np.all(own[:,pc.index('call_bb')]>0);action_count+=len(a)
  vm=dict(v);vm['d']=d.with_columns(pl.lit(0).alias('evidence'),pl.lit(-999).alias('evidence_rank'),pl.lit(0).alias('subtype'));mut=j.design(vm)
  for z,zmut in zip([own,bet,size],mut[:3]):np.testing.assert_array_equal(z,zmut)
  g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];pp=np.zeros((len(d),2));dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2))
  for f in range(4):
   extra=j.policy_fields(own,bet,size,f);np.testing.assert_array_equal(extra,np.load(j.ROOT/f'{fam}_extra_fold{f}.npz')['x']);assert np.isfinite(extra).all();np.testing.assert_allclose(extra[:,:4].sum(1),1,atol=2e-7);np.testing.assert_allclose(extra[:,4:8].sum(1),1,atol=2e-7);va=fv[g]==f;pp[g[va],actor[va]]=model(j.ROOT/f'{fam}_primary_fold{f}.cbm',np.column_stack([v['x'][va],extra[va]]));count+=1
  expect=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['bg_primary'].to_numpy();np.testing.assert_array_equal((pp*dw).sum(1),expect)
 z=saved.join(base,on=['pair_id','hand_id'],validate='1:1',suffix='_base');np.testing.assert_array_equal(z['bg_secondary'].to_numpy(),z['bg_secondary_base'].to_numpy());iso=states['coordinated_isolation']['d']['pair_id'].unique();q=z.filter(C('pair_id').is_in(iso.implode()));np.testing.assert_array_equal(q['bg_primary'].to_numpy(),q['bg_primary_base'].to_numpy());proof={'models_replayed':count,'score_error':0,'row_permutation_error':0,'raw_matched_bet_fold_actions_verified':action_count,'raw_and_policy_feature_replay_exact':True,'metadata_mutation_exact':True,'other_heads_unchanged':True};(j.ROOT/'verification.json').write_text(json.dumps(proof,indent=2));print(proof,flush=True)
if __name__=='__main__':main()
