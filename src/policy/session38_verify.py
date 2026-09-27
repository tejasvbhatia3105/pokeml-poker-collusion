import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session38_soft_bet_fold import ROOT,BASE,data,target,paired_features
C=pl.col
def main():
 d,a,cfg=data();saved=pl.read_parquet(ROOT/'fold_actions.parquet');assert a.equals(saved);extra,align=paired_features(d,a.reverse());assert extra.equals(pl.read_parquet(ROOT/'action_features.parquet'));assert align.equals(pl.read_parquet(ROOT/'alignment.parquet'));cfg2=json.load(open(ROOT/'config.json'));g=a['row'].to_numpy();fv=d['fold'].to_numpy();x=np.column_stack([a.select(cfg['action']).to_numpy(),d.select(cfg['hand']).to_numpy()[g]]);ex=extra.select(cfg2['paired_columns']).to_numpy();records=[]
 for kind in ['original_soft','paired_soft']:
  pp=np.zeros(len(d));xx=x if kind=='original_soft' else np.column_stack([x,ex])
  for f in range(4):
   y,e,_=target(d,f);tr=e[g];va=fv[g]==f;assert not (tr&va).any();assert y.sum()==y[g[tr]].sum();m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'primary_fold{f}.cbm'));pp[g[va]]=m.predict_proba(xx[va],thread_count=2)[:,1]
  q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(pp-q['bg_primary'].to_numpy()).max());assert err==0;base=pl.read_parquet(BASE/'event_oof.parquet');out=pl.read_parquet(ROOT/kind/'event_oof.parquet').join(base,on=['pair_id','hand_id'],suffix='_base',validate='1:1');assert float((out['bg_secondary']-out['bg_secondary_base']).abs().max())==0;other=out.filter(~C('pair_id').is_in(d['pair_id'].unique()));assert float((other['bg_primary']-other['bg_primary_base']).abs().max())==0;records.append({'kind':kind,'models_replayed':4,'primary_error':err,'nonsoft_primary_and_all_secondary_unchanged':True})
 report={'unique_fold_witness_hands':len(a),'raw_bet_fold_features_rebuild_reverse_order_exact':True,'models':records};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
