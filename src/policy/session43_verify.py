import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session43_pair_interactions as task
ROOT=task.ROOT;C=pl.col
def main():
 old=pl.read_parquet(ROOT/'action_features.parquet');new=task.build_actions(save=False);keys=['pair_id','hand_id'];assert new.sort(keys).equals(old.sort(keys));d=pl.read_parquet(ROOT/'window_features.parquet');cfg=json.load(open(ROOT/'config.json'));joint=cfg['joint_columns'];assert not any('style_' in c for c in joint);assert (d['label']>=0).all();assert d.filter((C('label')==1)&(C('n_ev_in')<1)).is_empty();assert d.group_by('pair_id').agg(C('fold').n_unique()).select(C('fold').max()).item()==1;fv=d['fold'].to_numpy();expected=d.select('pair_id','window').join(pl.read_parquet(ROOT/'oof.parquet'),on=['pair_id','window'],validate='1:1',maintain_order='left');errs={}
 for kind in ['residual_only','paired_context']:
  cols=cfg['base_columns'] if kind=='residual_only' else cfg['base_columns']+joint;x=d.select(cols).to_numpy();pp=np.zeros(len(d))
  for f in range(4):
   tr=fv!=f;va=fv==f;assert set(d['table_id'].to_numpy()[tr]).isdisjoint(set(d['table_id'].to_numpy()[va]));m=CatBoostClassifier();m.load_model(str(ROOT/f'{kind}_fold{f}.cbm'));assert list(m.classes_)==list(range(4));pp[va]=1-m.predict_proba(x[va],thread_count=2)[:,0]
  errs[kind]=float(abs(pp-expected[kind].to_numpy()).max());assert errs[kind]==0
                                                                            
 count_error=0;mean_error=0.
 for window,(lo,hi) in task.W.items():
  a=old.filter((C('time_index')>=lo)&(C('time_index')<hi));s=a.group_by('pair_id').agg(pl.len().alias('count'),C('response_equity').mean().alias('equity_mean'));q=d.filter(C('window')==window).join(s,on='pair_id',how='left',validate='1:1');count_error=max(count_error,int((q['joint_fold_count']-q['count'].fill_null(0)).abs().max()));mean_error=max(mean_error,float((q['joint_response_equity_mean']-q['equity_mean'].fill_null(-2)).abs().max()));assert q.select((C('joint_fold_count')<=C('policy_n_hands')).all()).item()
 assert count_error==0 and mean_error<1e-6;out={'all_action_features_raw_rebuild_exact':True,'paired_action_rows':len(old),'models_replayed':8,'risk_replay_errors':errs,'same_pair_all_windows_same_outer_fold':True,'unknown_pairs_never_negative':True,'new_features_exclude_phase_style_estimates':True,'window_action_count_error':count_error,'window_equity_mean_rebuild_error':mean_error,'positive_crops_all_retain_listed_evidence':True};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
