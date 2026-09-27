import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import log_loss
import session84_nested_joint_policy as s
from session82_83_verify import model
C=pl.col
def main():
 d,pc,native=s.reference_data();x=d.select(pc).to_numpy();y=d['action_class'].to_numpy();audit=json.load(open(s.ROOT/'reference_audit.json'));loss=[]
 for f,h in itertools.combinations(range(4),2):
  tr=(native!=f)&(native!=h);record=next(z for z in audit if z['excluded_folds']==[f,h]);assert set(record['training_tables'])==set(d.filter(pl.Series(tr))['table_id'].unique());assert record['training_actions']==int(tr.sum());assert not np.isin(native[tr],[f,h]).any();ap,sp=s.paths(f,h);m=CatBoostClassifier();m.load_model(str(ap));assert m.tree_count_==450;assert m.get_all_params()['random_seed']==310;va=~tr;pred=m.predict_proba(x[va],thread_count=3);reference=np.zeros_like(pred)
  for held in [f,h]:
   old=CatBoostClassifier();old.load_model(f'artifacts/policy/action_fold{held}.cbm');reference[native[va]==held]=old.predict_proba(x[va][native[va]==held],thread_count=3)
  reg=CatBoostRegressor();reg.load_model(str(sp));assert reg.tree_count_==350;assert reg.get_all_params()['random_seed']==510;assert np.isfinite(reg.predict(x[va][:256],thread_count=3)).all();loss.append({'excluded_folds':[f,h],'reference_heldout_action_logloss':float(log_loss(y[va],pred,labels=[0,1,2,3])),'original_reference_heldout_action_logloss':float(log_loss(y[va],reference,labels=[0,1,2,3])),'training_actions':int(tr.sum()),'validation_actions':int(va.sum())});print('verified reference',f,h,flush=True)
 del d,x,y,native
 states=s.old.state();saved=pl.read_parquet(s.ROOT/'event_oof.parquet');base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');count=0
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];raw=np.load(s.old.ROOT/f'{fam}_raw.npz');own,bet,size=raw['own'],raw['bet'],raw['size'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];af=fv[g];pp=np.zeros((len(d),2));dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2))
  for f in range(4):
   expected=np.zeros((len(a),15),np.float32)
   for h in range(4):
    if h==f:continue
    mask=(af==f)|(af==h);z=s.extra(own[mask],bet[mask],size[mask],f,h,pc);ix=np.flatnonzero(mask);expected[ix[af[mask]==h]]=z[af[mask]==h];expected[ix[af[mask]==f]]+=z[af[mask]==f]/3
   cached=np.load(s.ROOT/f'{fam}_extra_fold{f}.npz')['x'];np.testing.assert_array_equal(expected,cached);assert np.isfinite(expected).all();np.testing.assert_allclose(expected[:,:4].sum(1),1,atol=2e-7);np.testing.assert_allclose(expected[:,4:8].sum(1),1,atol=2e-7);va=af==f;pp[g[va],actor[va]]=model(s.ROOT/f'{fam}_primary_fold{f}.cbm',np.column_stack([v['x'][va],cached[va]]));count+=1
  expect=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['bg_primary'].to_numpy();np.testing.assert_array_equal((pp*dw).sum(1),expect)
 z=saved.join(base,on=['pair_id','hand_id'],suffix='_base',validate='1:1');np.testing.assert_array_equal(z['bg_secondary'].to_numpy(),z['bg_secondary_base'].to_numpy());iso=states['coordinated_isolation']['d']['pair_id'].unique();z=z.filter(C('pair_id').is_in(iso.implode()));np.testing.assert_array_equal(z['bg_primary'].to_numpy(),z['bg_primary_base'].to_numpy());proof={'reference_models_checked':12,'reference_exclusion_reconstruction_exact':True,'evidence_heads_replayed':count,'feature_cache_replay_error':0,'evidence_score_replay_error':0,'row_permutation_error':0,'other_heads_unchanged':True,'reference_heldout_quality':loss};(s.ROOT/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
