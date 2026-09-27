import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data,targets
C=pl.col;ROOT=Path('artifacts/evidence_session31_conditional_events')
def main():
 d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.;audit=[]
 with threadpool_limits(limits=2):
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    a,b,ea,eb,va=targets(d,f,fam);e=ea&eb&~a;assert not (e&va).any();assert (b&e).sum()==(b&eb).sum();m=CatBoostClassifier();m.load_model(str(ROOT/f'secondary_{fam}_fold{f}.cbm'));cp=m.predict_proba(X[va],thread_count=2)[:,1];h=joblib.load(ROOT/f'secondary_{fam}_fold{f}.joblib');hp=h.predict_proba(X[va])[:,1];err=max(err,float(abs(cp-q['cat_conditional_secondary'].to_numpy()[va]).max()),float(abs(hp-q['hist_conditional_secondary'].to_numpy()[va]).max()));audit.append({'fold':f,'family':fam,'conditional_rows':int(e.sum()),'positive_labels_preserved':True,'validation_overlap':0})
 identities={}
 for first,second,conditional in [('bg_primary','bg_secondary','cat_conditional_secondary'),('new_hist_primary','new_hist_secondary','hist_conditional_secondary')]:
  p=q[first].to_numpy();s=q[second].to_numpy();v=q[conditional].to_numpy();delta=float(abs(s-(1-p)*v).max());assert delta==0.;assert np.max(p+s)<=1+1e-12;identities[first]={'factorization_error':delta,'max_event_mass':float(np.max(p+s))}
 assert err<1e-12;report={'conditional_models_replayed':24,'max_conditional_probability_error':err,'factorization':identities,'targets':audit};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
