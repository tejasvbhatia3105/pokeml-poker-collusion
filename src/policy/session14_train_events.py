import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session14_outcomes')
def main():
 extra=pl.read_parquet(ROOT/'hand_features.parquet');d=hand_data().join(extra,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');old=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];start=time.time()
 for arm in os.environ.get('OUTCOME_ARMS','factual,all').split(','):
  root=ROOT/arm;root.mkdir(exist_ok=True);added=[c for c in extra.columns if c.startswith('contrast_')]
  if arm=='factual':added=[c for c in added if '_prediction_' in c or '_residual_' in c]
  cols=old+added;(root/'columns.json').write_text(json.dumps(cols,indent=2));X=d.select(cols).to_numpy();assert np.isfinite(X).all();pred=np.zeros((len(d),2));audit=[]
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    a,b,ea,eb,va=targets(d,f,fam)
    for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
     assert not np.any(e&va);path=root/f'event{k}_{fam}_fold{f}.cbm';m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False)
     if path.exists():m.load_model(str(path))
     else:m.fit(X[e],y[e].astype(int));m.save_model(str(path))
     pred[va,k-1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'family':fam,'head':k,'training_rows':int(e.sum()),'validation_overlap':int((e&va).sum())})
    print('outcome event',arm,f,fam,round(time.time()-start,1),flush=True)
  d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(root/'event_oof.parquet');(root/'audit.json').write_text(json.dumps(audit,indent=2));assemble(root)
if __name__=='__main__':main()
