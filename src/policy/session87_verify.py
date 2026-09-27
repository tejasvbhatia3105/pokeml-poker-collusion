import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl,torch
import session87_history_evidence as s
from session82_83_verify import model
C=pl.col
def main():
 torch.set_num_threads(3);states=s.old.state();count=mutations=0;base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet')
 for kind in ['summary','history']:
  saved=pl.read_parquet(s.ROOT/kind/'event_oof.parquet')
  for fam in ['directed_transfer','soft_play']:
   v=states[fam];d,a=v['d'],v['a'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];pp=np.zeros((len(d),2));dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2));mut=d.with_columns(pl.when(C('fold')>=0).then(1-C('evidence')).otherwise(C('evidence')).alias('dummy'))
   for f in range(4):
    ex=s.features(fam,v,f,kind);np.testing.assert_array_equal(ex,np.load(s.ROOT/kind/f'{fam}_extra_fold{f}.npz')['x']);assert np.isfinite(ex).all();np.testing.assert_allclose(ex[:,:4].sum(1),1,atol=2e-7);np.testing.assert_allclose(ex[:,4:8].sum(1),1,atol=2e-7);va=fv[g]==f;pp[g[va],actor[va]]=model(s.ROOT/kind/f'{fam}_primary_fold{f}.cbm',np.column_stack([v['x'][va],ex[va]]));count+=1;mut=d.with_columns(pl.when(C('fold')==f).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when(C('fold')==f).then(999).otherwise(C('evidence_rank')).alias('evidence_rank'))
    if fam=='directed_transfer':
     y,tr,_=s.old.labels(d,a,f);ym,tm,_=s.old.labels(mut,a,f)
    else:y,tr,_=s.old.target(d,f);ym,tm,_=s.old.target(mut,f)
    np.testing.assert_array_equal(tr,tm);np.testing.assert_array_equal(y[tr],ym[tr]);mutations+=1
   expected=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['bg_primary'].to_numpy();np.testing.assert_array_equal((pp*dw).sum(1),expected)
  z=saved.join(base,on=['pair_id','hand_id'],validate='1:1',suffix='_base');np.testing.assert_array_equal(z['bg_secondary'].to_numpy(),z['bg_secondary_base'].to_numpy());iso=states['coordinated_isolation']['d']['pair_id'].unique();z=z.filter(C('pair_id').is_in(iso.implode()));np.testing.assert_array_equal(z['bg_primary'].to_numpy(),z['bg_primary_base'].to_numpy())
 proof={'event_models_replayed':count,'event_score_error':0,'representation_feature_cache_error':0,'heldout_target_mutation_checks':mutations,'mutation_error':0,'other_heads_unchanged':True,'encoder_fit_scope':'outer training pools only; same frozen basis for event train and validation; not inner-cross-fitted'};(s.ROOT/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
