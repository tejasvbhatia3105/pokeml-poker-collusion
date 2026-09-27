import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session52_call_comparison import ROOT,data,features,targets,OLD,C
def main():
 d,a,cfg=data();XX=features(d,a);np.testing.assert_array_equal(XX,np.load(ROOT/'features.npz')['x']);np.testing.assert_array_equal(XX,features(d.with_columns((1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank')),a));AX=np.load(OLD/'actor_features.npz')['hand'];hx=d.select(cfg['hand']).to_numpy();fv=d['fold'].to_numpy();err=0;old_error=0;models=0
 for f in range(4):
  y,tr,donor=targets(d,f);vi=np.flatnonzero(fv==f);assert not tr[vi].any();old=CatBoostClassifier();old.load_model(str(OLD/'oriented'/f'event2_directed_transfer_fold{f}.cbm'))
  for kind in ['control','comparison']:
   root=ROOT/kind;ref=d.select('pair_id','hand_id').join(pl.read_parquet(root/'conditional_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');m=CatBoostClassifier();m.load_model(str(root/f'secondary_fold{f}.cbm'));models+=1
   for actor in range(2):
    baseline=np.column_stack([hx[vi],AX[vi,actor]]);x=np.column_stack([baseline,XX[vi,actor]]) if kind=='comparison' else baseline;p=m.predict_proba(x,thread_count=2)[:,1];err=max(err,float(abs(p-ref[f'actor{actor}'].to_numpy()[vi]).max()))
    if kind=='control':old_error=max(old_error,float(abs(p-old.predict_proba(baseline,thread_count=2)[:,1]).max()))
 base=pl.read_parquet('artifacts/evidence_session50_matchup/current/event_oof.parquet');control=pl.read_parquet(ROOT/'control/event_oof.parquet').join(base,on=['pair_id','hand_id'],suffix='_base',validate='1:1');control_error=max(float((control[c]-control[c+'_base']).abs().max()) for c in ['bg_primary','bg_secondary']);assert err==0 and old_error==0 and control_error==0
 new=pl.read_parquet(ROOT/'comparison/event_oof.parquet').join(base,on=['pair_id','hand_id'],suffix='_base',validate='1:1');assert (new['bg_primary']==new['bg_primary_base']).all();outside=new.filter(~C('pair_id').is_in(d['pair_id'].unique()));assert (outside['bg_secondary']==outside['bg_secondary_base']).all();audit=json.load(open(ROOT/'feature_audit.json'));report={'saved_models_replayed':models,'probability_error':err,'original25_secondary_control_error':old_error,'all_R32_control_event_probabilities_error':control_error,'feature_cache_replay_exact':True,'labels_and_ranks_do_not_change_features':True,'other_heads_unchanged':True,'raw_action_classes_checked':audit['raw_action_classes_checked'],'allin_calls_checked':audit['allin_calls_checked'],'call_cache_complete':audit['call_cache_exact_complete_against_raw_policy_actions']};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
