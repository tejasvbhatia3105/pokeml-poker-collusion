import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session96_player_context_evidence as s
C=pl.col
def main():
 ms=s.models();states=s.old.state();raw_actions=0;pairchecks=0;featurechecks=0;mutationchecks=0;models=0;shared_records=[];labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');nonstyle=[c for c in s.p.s.PC if not c.startswith(('style_','local_style_'))];base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet')
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];savedq=pl.read_parquet(s.ROOT/f'{fam}_queries.parquet');_,align=s.paired_features(d,a);oldalign=pl.read_parquet(s.ROOT/f'{fam}_alignment.parquet');np.testing.assert_array_equal(align.to_numpy(),oldalign.to_numpy());chosen=set(sorted(savedq['pair_id'].unique().to_list())[::max(1,savedq['pair_id'].n_unique()//8)][:8])
  for (table,),g in d.group_by('table_id'):
   src=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');local=savedq.join(g.select('pair_id').unique(),on='pair_id',how='semi');srcq=src[local['source_row'].to_numpy()];np.testing.assert_array_equal(local.select('hand_id','player_id',*nonstyle).to_numpy(),srcq.select('hand_id','player_id',*nonstyle).to_numpy());raw_actions+=len(local);seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(src.select('hand_id').unique(),on='hand_id',how='semi');hands={pid:set(z['hand_id']) for (pid,),z in seats.group_by('player_id')}
   for (pid,),dg in g.group_by('pair_id'):
    one,two=labs.filter(C('pair_id')==pid).select('player_1','player_2').row(0);shared=hands[one]&hands[two];assert shared==set(dg['hand_id']);shared_records.append({'pair_id':pid,'raw_shared':len(shared),'cached_shared':len(dg),'missing':0,'extra':0})
    if pid not in chosen:continue
    old=local.filter(C('pair_id')==pid).sort('query_row');keys=old.select('action_row','pair_id','hand_id','action_no',C('player_id').alias('expected_player'),'role','query_row');q,h=s.pair_input(src,keys,shared);np.testing.assert_array_equal(q.select(s.p.s.PC).to_numpy(),old.select(s.p.s.PC).to_numpy());native=int(dg['fold'][0]);ex=s.policy_features(q,h,native,ms);indices=q.filter(C('role')==0)['action_row'].to_numpy();mut=src.with_columns(pl.when(C('hand_id').is_in(pl.Series(list(shared)).implode())).then((C('action_class')+1)%4).otherwise(C('action_class')).alias('action_class'));mq,mh=s.pair_input(mut,keys,shared);np.testing.assert_array_equal(q.select(s.p.s.PC).to_numpy(),mq.select(s.p.s.PC).to_numpy());np.testing.assert_array_equal(h.to_numpy(),mh.to_numpy());mex=s.policy_features(mq,mh,native,ms)
    for (f,k),x in ex.items():np.testing.assert_array_equal(x,np.load(s.ROOT/k/f'{fam}_extra_fold{f}.npz')['x'][indices]);np.testing.assert_array_equal(x,mex[f,k]);featurechecks+=1
    pairchecks+=1
  fv=v['fv'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2))
  for kind in s.KINDS:
   saved=pl.read_parquet(s.ROOT/kind/'event_oof.parquet');pp=np.zeros((len(d),2))
   for f in range(4):
    dm=d.with_columns(*[pl.when(C('fold')==f).then(pl.lit(val)).otherwise(C(col)).alias(col) for col,val in [('evidence',0),('evidence_rank',-999),('subtype',-999)]])
    if fam=='directed_transfer':y,tr,va=s.old.labels(d,a,f);ym,tm,vm=s.old.labels(dm,a,f)
    else:y,e,_=s.old.target(d,f);ym,em,_=s.old.target(dm,f);tr=e[g];tm=em[g];va=fv[g]==f;vm=va
    for aa,bb in [(y,ym),(tr,tm),(va,vm)]:np.testing.assert_array_equal(aa,bb)
    assert not(tr&va).any();mutationchecks+=1;x=np.column_stack([v['x'],np.load(s.ROOT/kind/f'{fam}_extra_fold{f}.npz')['x']]);m=CatBoostClassifier();m.load_model(str(s.ROOT/kind/f'{fam}_primary_fold{f}.cbm'));p=m.predict_proba(x[va],thread_count=2)[:,1];np.testing.assert_array_equal(p,m.predict_proba(x[va][::-1],thread_count=2)[:,1][::-1]);pp[g[va],actor[va]]=p;models+=1
   expected=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['bg_primary'].to_numpy();np.testing.assert_array_equal((pp*dw).sum(1),expected);z=saved.join(base,on=['pair_id','hand_id'],validate='1:1',suffix='_base');np.testing.assert_array_equal(z['bg_secondary'].to_numpy(),z['bg_secondary_base'].to_numpy());iso=states['coordinated_isolation']['d']['pair_id'].unique();zz=z.filter(C('pair_id').is_in(iso.implode()));np.testing.assert_array_equal(zz['bg_primary'].to_numpy(),zz['bg_primary_base'].to_numpy())
 report={'raw_matched_query_actions':raw_actions,'raw_seat_shared_pair_checks':len(shared_records),'missing_or_extra_shared_hands':0,'pair_context_rebuilds_and_shared_outcome_mutations':pairchecks,'policy_feature_blocks_replayed':featurechecks,'models_replayed':models,'heldout_target_mutations':mutationchecks,'final_event_score_error':0,'query_permutation_error':0,'unchanged_secondary_and_isolation_heads':True};(s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));(s.ROOT/'raw_shared_audit.json').write_text(json.dumps({'pairs':len(shared_records),'missing':0,'extra':0,'records':shared_records},indent=2));print(report)
if __name__=='__main__':main()
