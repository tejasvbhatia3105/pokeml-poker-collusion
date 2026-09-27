import json
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session69_pressure_relabel as relabel
from session57_isolation_pressure import data,BASE,noisy_or
from session67_isolation_types import design
from session59_pressure_equity import ROOT as PRECISE
def main():
 full,d,a,ac=data();relabel.TYPE_X,tc=design(d,a,ac);g=a['row'].to_numpy();fv=d['fold'].to_numpy();cfg=json.load(open(PRECISE/'config.json'));x=np.column_stack([a.select(ac).to_numpy(),d.select(cfg['hand_columns']).to_numpy()[g],np.load(PRECISE/'features.npz')['x']]);maxactive=relabel.TYPE_X[:,tc.index('players_active_max')];sub=d['subtype'].to_numpy();target_checks=0
 for f in range(4):
  want=relabel.targets(full,f,'coordinated_isolation');changed=full.with_columns(*[pl.when(pl.col('fold')==f).then(pl.lit(value)).otherwise(pl.col(col)).alias(col) for col,value in [('evidence',0),('evidence_rank',-123),('subtype',2),('time',-999)]]);got=relabel.targets(changed,f,'coordinated_isolation')
  for u,v in zip(want,got):np.testing.assert_array_equal(u,v)
  target_checks+=1
 for n in [69,70]:
  root=Path(f'artifacts/evidence_session{n}_'+('pressure_relabel' if n==69 else 'active_pressure'));saved=d.select('pair_id','hand_id').join(pl.read_parquet(root/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=perm=0;models=0;zero_audit=[]
  for f in range(4):
   for k in range(2):
    level=int(np.unique(maxactive[(fv!=f)&(sub==k+1)])[0]);support=np.ones(len(a),bool) if n==69 else a['players_active'].to_numpy()==level;va=(fv[g]==f)&support
    for em in range(3):
     m=CatBoostClassifier();m.load_model(str(root/f'event{k+1}_fold{f}_em{em}.cbm'));p=m.predict_proba(x[va],thread_count=2)[:,1];hp=noisy_or(p,g[va],len(d));hp2=noisy_or(m.predict_proba(x[va][::-1],thread_count=2)[:,1],g[va][::-1],len(d));perm=max(perm,float(abs(hp-hp2).max()));models+=1
     if em==2:
      err=max(err,float(abs(hp[fv==f]-saved[['bg_primary','bg_secondary'][k]].to_numpy()[fv==f]).max()));cnt=np.bincount(g[support],minlength=len(d));assert (hp[(fv==f)&(cnt==0)]==0).all();zero_audit.append({'fold':f,'head':k+1,'unsupported_probability_zero':True})
  assert err==0 and perm<1e-12
  z=pl.read_parquet(root/'event_oof.parquet').join(pl.read_parquet(BASE/'event_oof.parquet'),on=['pair_id','hand_id'],suffix='_base',validate='1:1').filter(~pl.col('pair_id').is_in(d['pair_id'].unique().implode()))
  for c in ['bg_primary','bg_secondary']:np.testing.assert_array_equal(z[c].to_numpy(),z[c+'_base'].to_numpy())
  report={'saved_models_replayed':models,'event_probability_error':err,'action_permutation_error':perm,'outer_heldout_target_mutations':target_checks,'other_families_unchanged':True,'raw_feature_basis':'57/59/67 independent raw verifications','support_checks':zero_audit};(root/'verification.json').write_text(json.dumps(report,indent=2));print(n,json.dumps(report),flush=True)
if __name__=='__main__':main()
