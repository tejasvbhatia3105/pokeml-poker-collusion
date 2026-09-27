import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session97_player_pressure as s
C=pl.col
def main():
 full,d,a,ac=s.data();qcache=pl.read_parquet(s.ROOT/'queries.parquet');ms=s.s.models();nonstyle=[c for c in s.s.p.s.PC if not c.startswith(('style_','local_style_'))];selected=set(sorted(d['pair_id'].unique().to_list())[::11][:8]);labels=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');raw=0;sharedchecks=0;pairchecks=0;fields=0
 for (table,),dg in d.group_by('table_id'):
  src=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');q=qcache.join(dg.select('pair_id').unique(),on='pair_id',how='semi');sq=src[q['source_row'].to_numpy()];np.testing.assert_array_equal(q.select('hand_id','player_id',*nonstyle).to_numpy(),sq.select('hand_id','player_id',*nonstyle).to_numpy());assert (sq['action_class']==3).all();raw+=len(q);seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(src.select('hand_id').unique(),on='hand_id',how='semi');hands={pid:set(z['hand_id']) for (pid,),z in seats.group_by('player_id')}
  for (pid,),g in dg.group_by('pair_id'):
   one,two=labels.filter(C('pair_id')==pid).select('player_1','player_2').row(0);shared=hands[one]&hands[two];assert shared==set(g['hand_id']);sharedchecks+=1
   if pid not in selected:continue
   old=q.filter(C('pair_id')==pid).sort('query_row');keys=old.select('action_row','pair_id','hand_id','action_no',C('player_id').alias('expected_player'),'role','query_row');qq,h=s.s.pair_input(src,keys,shared);np.testing.assert_array_equal(qq.select(s.s.p.s.PC).to_numpy(),old.select(s.s.p.s.PC).to_numpy());native=int(g['fold'][0]);ex=s.policy_features(qq,h,native,ms);mut=src.with_columns(pl.when(C('hand_id').is_in(pl.Series(list(shared)).implode())).then((C('action_class')+1)%4).otherwise(C('action_class')).alias('action_class'));mq,mh=s.s.pair_input(mut,keys,shared);np.testing.assert_array_equal(qq.select(s.s.p.s.PC).to_numpy(),mq.select(s.s.p.s.PC).to_numpy());np.testing.assert_array_equal(h.to_numpy(),mh.to_numpy());mex=s.policy_features(mq,mh,native,ms);ix=qq['action_row'].to_numpy()
   for (f,k),x in ex.items():np.testing.assert_array_equal(x,np.load(s.ROOT/k/f'extra_fold{f}.npz')['x'][ix]);np.testing.assert_array_equal(x,mex[f,k]);fields+=1
   pairchecks+=1
 g=a['row'].to_numpy();fv=d['fold'].to_numpy();hc=json.load(open('artifacts/evidence_session59_pressure_equity/config.json'))['hand_columns'];xx=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x']]);models=0;mutations=0;fm=full['behavior_family'].to_numpy()=='coordinated_isolation'
 for k in s.KINDS:
  saved=pl.read_parquet(s.ROOT/k/'event_oof.parquet');pred=np.zeros((len(d),2))
  for f in range(4):
   x=np.column_stack([xx,np.load(s.ROOT/k/f'extra_fold{f}.npz')['x']]);va=fv[g]==f;labels0=s.targets(full,f,'coordinated_isolation');dm=full.with_columns(*[pl.when(C('fold')==f).then(pl.lit(val)).otherwise(C(col)).alias(col) for col,val in [('evidence',0),('evidence_rank',-999),('subtype',-999)]]);labels1=s.targets(dm,f,'coordinated_isolation')
   for z,zz in zip(labels0[:4],labels1[:4]):np.testing.assert_array_equal(z,zz)
   for head in range(2):
    tr=labels0[2+head][fm][g];assert not(tr&va).any();mutations+=1
    for em in range(3):
     m=CatBoostClassifier();m.load_model(str(s.ROOT/k/f'head{head+1}_fold{f}_em{em}.cbm'));p=m.predict_proba(x[va],thread_count=2)[:,1];np.testing.assert_array_equal(p,m.predict_proba(x[va][::-1],thread_count=2)[:,1][::-1]);assert np.isfinite(p).all();models+=1
    hp=s.noisy_or(p,g[va],len(d));pred[fv==f,head]=hp[fv==f]
  expected=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left').select('bg_primary','bg_secondary').to_numpy();np.testing.assert_array_equal(pred,expected);base=pl.read_parquet(s.s.ROOT/k/'event_oof.parquet');other=saved.join(d.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='anti').join(base,on=['pair_id','hand_id'],suffix='_base',validate='1:1');np.testing.assert_array_equal(other.select('bg_primary','bg_secondary').to_numpy(),other.select('bg_primary_base','bg_secondary_base').to_numpy())
 out={'raw_pressure_actions':raw,'raw_shared_pair_seat_checks':sharedchecks,'shared_hand_differences':0,'context_pair_rebuilds_and_outcome_mutations':pairchecks,'policy_feature_blocks_replayed':fields,'saved_models_replayed':models,'target_exclusion_checks':mutations,'final_event_probability_error':0,'query_permutation_error':0,'nonisolation_heads_preserve96':True};(s.ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
