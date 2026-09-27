import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session41_isolation_bet_fold import ROOT,BASE,data,designs,paired_features,targets
C=pl.col
def main():
 full,d,a,ac=data();saved=pl.read_parquet(ROOT/'fold_actions.parquet');keys=['pair_id','hand_id','actor','action_no'];assert a.drop('action_row').sort(keys).equals(saved.drop('action_row').sort(keys));a=saved;extra,align=paired_features(d,a.reverse());assert extra.equals(pl.read_parquet(ROOT/'action_features.parquet'));assert align.equals(pl.read_parquet(ROOT/'alignment.parquet'));xs,_,_=designs(d,a,ac,extra);dv=d['fold'].to_numpy();familymask=full['behavior_family'].to_numpy()=='coordinated_isolation';records=[]
 for kind,x in xs.items():
  pp=np.zeros((len(d),2))
  for f in range(4):
   p1,p2,e1,e2,_=targets(full,f,'coordinated_isolation');es=np.column_stack([e1,e2])[familymask];va=dv==f
   for k in range(2):
    assert not (es[:,k]&va).any();m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'event{k+1}_fold{f}.cbm'));pp[va,k]=m.predict_proba(x[va],thread_count=2)[:,1]
  q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(pp-q.select('bg_primary','bg_secondary').to_numpy()).max());assert err==0;out=pl.read_parquet(ROOT/kind/'event_oof.parquet').join(pl.read_parquet(BASE/'event_oof.parquet'),on=['pair_id','hand_id'],suffix='_base',validate='1:1').filter(~C('pair_id').is_in(d['pair_id'].unique()));assert max(float((out[c]-out[c+'_base']).abs().max()) for c in ['bg_primary','bg_secondary'])==0;records.append({'kind':kind,'models_replayed':8,'probability_error':err,'other_family_heads_unchanged':True})
 out={'paired_fold_hands':len(a),'all_isolation_hands':len(d),'no_fold_hands_preserved':len(d)-len(a),'raw_features_rebuild_reverse_order_exact':True,'models':records};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
