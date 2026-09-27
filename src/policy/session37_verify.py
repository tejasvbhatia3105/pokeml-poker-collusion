import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session37_bet_fold import ROOT,EXACT,OLD,paired_features,labels
from session8_data import hand_data
C=pl.col
def main():
 d=hand_data().filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');a=pl.read_parquet(EXACT/'fold_actions.parquet').with_row_index('action_row');extra,align=paired_features(d,a.reverse());saved=pl.read_parquet(ROOT/'action_features.parquet');assert extra.equals(saved);assert align.equals(pl.read_parquet(ROOT/'alignment.parquet'));assert (align['bet_action_no']<align['fold_action_no']).all();cfg=json.load(open(ROOT/'config.json'));g=a['row'].to_numpy();actor=a['actor'].to_numpy();ax=a.select(cfg['fold_columns']).to_numpy();hx=d.select(cfg['hand_columns']).to_numpy()[g];ex=extra.select(cfg['paired_columns']).to_numpy();dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();report=[]
 for kind in ['paired_action','paired_hand']:
  x=np.column_stack([ax,ex] if kind=='paired_action' else [ax,hx,ex]);pp=np.zeros((len(d),2))
  for f in range(4):
   y,tr,va=labels(d,a,f);assert not (tr&va).any();m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'primary_fold{f}.cbm'));pp[g[va],actor[va]]=m.predict_proba(x[va],thread_count=2)[:,1]
  q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'conditional_primary.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(pp-q.select('actor0_primary','actor1_primary').to_numpy()).max());q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet').select('pair_id','hand_id','bg_primary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');weighted=float(abs((pp*dw).sum(1)-q['bg_primary'].to_numpy()).max());assert max(err,weighted)==0;report.append({'kind':kind,'models_replayed':4,'conditional_error':err,'actor_aggregation_error':weighted})
 out={'raw_action_pairs':len(a),'bet_precedes_fold_all':True,'partner_is_last_aggressor_all':True,'no_intervening_raise_all':True,'raw_feature_rebuild_and_reverse_order_exact':True,'models':report};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
