import os,json,sys
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session34_policy_rollout import roots_for_table,simulate,aggregate
from session34_policy_state import vector
C=pl.col;ROOT=Path('artifacts/evidence_session34_river')
def main():
 d=hand_data();f=pl.read_parquet(ROOT/'hand_features.parquet');assert f.shape==(len(d),290);audit=json.load(open(ROOT/'rollout_audit.json'));assert max(r['max_steps'] for r in audit)<128 and max(r['max_net_sum_error'] for r in audit)<1e-8;path=ROOT/'feature_verification.json'
 if not path.exists():
  labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');folds=json.load(open('artifacts/policy/table_folds.json'));reports=[]
  for rec in sorted(audit,key=lambda r:-r['root_actions'])[:2]:
   table=rec['table_id'];fold=folds[table];q=d.filter(C('table_id')==table).select('pair_id','hand_id').join(labs,on='pair_id',validate='m:1');policy=CatBoostClassifier();policy.load_model(f'artifacts/policy/action_fold{fold}.cbm');size=CatBoostClassifier();size.load_model(f'artifacts/evidence_session22_size_density/density_fold{fold}.cbm');meta=json.load(open(f'artifacts/evidence_session22_size_density/metadata_fold{fold}.json'));roots=roots_for_table(table,q);values,info=simulate(roots,policy,size,meta);z=aggregate(roots,values,q);cols=[c for c in z.columns if c.startswith('rollout_')];want=z.select('pair_id','hand_id').join(f,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(z.select(cols).to_numpy()-want.select(cols).to_numpy()).max());assert err==0.
   chosen=[r for r in roots if r['forced_class']<3][:4];v,_=simulate(chosen,policy,size,meta);rev,_=simulate(chosen[::-1],policy,size,meta);perm=float(abs(v-rev[::-1]).max());assert perm==0.;extended,_=simulate(chosen,policy,size,meta,reps=64);assert np.array_equal(v,extended[:,:,:32]);matched=0;u=np.random.default_rng(3434).random((32,128,2))
   for i,r in enumerate(chosen):
    st=r['state'];j=r['own'];x=vector(st,j,r['templates'][j],r['bb']);p=policy.predict_proba(x[None],thread_count=2)[0];call=st.to_call(j);legal=np.array([call>0,call==0,call>0,st.remaining[j]>call and st.raise_right[j] and np.any(st.alive&(st.remaining>0)&(np.arange(6)!=j))]);p=np.maximum(p,1e-12)*legal;p/=p.sum();k=(u[:,0,0,None]>p.cumsum()[None]).sum(1).clip(0,3);same=k==r['forced_class'];matched+=int(same.sum());assert np.array_equal(v[i,0,same],v[i,1,same]);assert np.array_equal(v[i,2,same],v[i,3,same])
   mc=float(abs(v.mean(2)-extended.mean(2)).max()) if len(chosen) else 0.;reports.append({'table':table,'feature_rebuild_error':err,'permutation_error':perm,'same_focal_action_zero_delta_checks':matched,'replicate_prefix_32_of64_exact':True,'sample_32_vs64_mean_payoff_max_difference_chips':mc,**info})
  report={'engine_replay':json.load(open(ROOT/'replay.json')),'policy_input_replay':json.load(open(ROOT/'policy_state_replay.json')),'global_trajectories':sum(r['trajectories'] for r in audit),'global_simulated_actions':sum(r['simulated_actions'] for r in audit),'max_steps':max(r['max_steps'] for r in audit),'sampled_rollout_replays':reports,'caveat':'32-replicate estimates have sampling uncertainty; sample64 comparison is not a global precision guarantee'};path.write_text(json.dumps(report,indent=2))
 if '--features-only' in sys.argv:print(path.read_text());return
 q=d.join(f,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');models={}
 for kind in ['checkcall','learned']:
  cols=json.load(open(ROOT/kind/'columns.json'));X=q.select(cols).to_numpy();saved=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.;count=0
  for fold in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    va=(d['fold'].to_numpy()==fold)&(d['behavior_family'].to_numpy()==fam)
    for k in [1,2]:
     m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'event{k}_{fam}_fold{fold}.cbm'));p=m.predict_proba(X[va],thread_count=2)[:,1];err=max(err,float(abs(p-saved['bg_primary' if k==1 else 'bg_secondary'].to_numpy()[va]).max()));count+=1
  assert err<1e-12;models[kind]={'models':count,'max_error':err}
 report={'features':json.loads(path.read_text()),'models':models};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
