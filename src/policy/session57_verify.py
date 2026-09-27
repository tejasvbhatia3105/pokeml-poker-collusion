import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session57_isolation_pressure import data,ROOT,BASE,targets,noisy_or,C
def main():
 full,d,a,ac=data();cfg=json.load(open(ROOT/'config.json'));assert ac==cfg['action_columns'];assert a.equals(pl.read_parquet(ROOT/'pressure_actions.parquet'));g=a['row'].to_numpy();fv=d['fold'].to_numpy();hx=d.select(cfg['hand_columns']).to_numpy();ax=a.select(ac).to_numpy();xs={'action':ax,'action_hand':np.column_stack([ax,hx[g]])};mask=full['behavior_family'].to_numpy()=='coordinated_isolation';models=0;err=perm=0
 mutated=d.with_columns((1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'),pl.lit(99).alias('subtype'));np.testing.assert_array_equal(hx,mutated.select(cfg['hand_columns']).to_numpy())
 for f in range(4):
  original=targets(full,f,'coordinated_isolation');changed=full.with_columns(pl.when(C('fold')==f).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when(C('fold')==f).then(999).otherwise(C('evidence_rank')).alias('evidence_rank'),pl.when(C('fold')==f).then(99).otherwise(C('subtype')).alias('subtype'));other=targets(changed,f,'coordinated_isolation')
  for i in range(4):np.testing.assert_array_equal(original[i],other[i])
  va=fv[g]==f
  for kind,x in xs.items():
   saved=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
   for k in range(2):
    tr=original[2+k][mask][g];assert not(tr&va).any()
    for em in range(3):
     m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'event{k+1}_fold{f}_em{em}.cbm'));models+=1;p=m.predict_proba(x[va],thread_count=2)[:,1];hp=noisy_or(p,g[va],len(d));hp2=noisy_or(m.predict_proba(x[va][::-1],thread_count=2)[:,1],g[va][::-1],len(d));perm=max(perm,float(abs(hp-hp2).max()))
     if em==2:err=max(err,float(abs(hp[fv==f]-saved[['bg_primary','bg_secondary'][k]].to_numpy()[fv==f]).max()))
 assert err==0 and perm<1e-12
 labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');total=allin=0
 for (table,),q in d.group_by('table_id'):
  query=q.select('pair_id','hand_id').join(labs,on='pair_id');mapping=pl.concat([query.select('pair_id','hand_id',C('player_1').alias('player_id'),C('player_2').alias('partner')),query.select('pair_id','hand_id',C('player_2').alias('player_id'),C('player_1').alias('partner'))]);needed=q.select('hand_id').unique();pc=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').with_columns(C('action_no').cast(pl.Int64)).join(needed,on='hand_id',how='semi');raw=pl.read_parquet(f'artifacts/compact/actions/table_id={table}/*.parquet').join(needed,on='hand_id',how='semi');cc=pl.when(C('amount')>C('to_call')).then(3).when(C('action')=='fold').then(0).when(C('action')=='check').then(1).otherwise(2);raw=raw.with_columns(cc.alias('correct_class'));z=pc.select('hand_id','action_no','action_class').join(raw.select('hand_id',C('action_no').cast(pl.Int64),'correct_class','action'),on=['hand_id','action_no'],validate='1:1');assert (z['action_class']==z['correct_class']).all();total+=len(z);allin+=len(z.filter((C('action')=='all_in')&(C('correct_class')==3)))
  folds=raw.filter(C('action')=='fold').select('hand_id',C('player_id').alias('partner'),C('action_no').alias('partner_fold_no'));expected=pc.join(mapping,on=['hand_id','player_id']).join(folds,on=['hand_id','partner'],how='left',validate='m:1').filter((C('action_class')==3)&~(C('last_aggressor')==C('partner')).fill_null(False)&(C('players_active')>=3)&(C('partner_fold_no').is_null()|(C('partner_fold_no')>=C('action_no'))));cached=a.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');keys=['pair_id','hand_id','street_no','action_no'];assert set(expected.select(keys).iter_rows())==set(cached.select(keys).iter_rows())
 base=pl.read_parquet(BASE/'event_oof.parquet');support=d.select('pair_id','hand_id').with_columns(pl.Series('supported',np.bincount(g,minlength=len(d))>0))
 for kind in ['support_only',*xs]:
  z=pl.read_parquet(ROOT/kind/'event_oof.parquet').join(base,on=['pair_id','hand_id'],suffix='_base',validate='1:1').join(support,on=['pair_id','hand_id'],how='left',validate='1:1');outside=z.filter(C('supported').is_null())
  for col in ['bg_primary','bg_secondary']:
   np.testing.assert_array_equal(outside[col].to_numpy(),outside[col+'_base'].to_numpy());assert (z.filter(C('supported')==False)[col]==0).all()
   if kind=='support_only':np.testing.assert_array_equal(z.filter(C('supported'))[col].to_numpy(),z.filter(C('supported'))[col+'_base'].to_numpy())
 out={'saved_models_replayed':models,'event_probability_error':err,'action_order_permutation_error':perm,'heldout_target_label_rank_subtype_mutation_cases':4,'hand_features_label_invariant':True,'raw_action_classes_checked':total,'allin_raises_checked':allin,'pressure_cache_exact_complete_against_raw_actions':True,'support_only_control_exact':True,'other_families_unchanged':True};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
