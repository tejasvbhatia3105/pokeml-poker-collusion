import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session13_actor')
def main():
 d=hand_data();extra=pl.read_parquet(ROOT/'hand_features.parquet');d=d.join(extra,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');old=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];start=time.time()
 for arm in os.environ.get('ACTOR_ARMS','global,adapted').split(','):
  root=ROOT/arm;root.mkdir(exist_ok=True);prefix='global' if arm.startswith('global') else 'adapted';added=[c for c in extra.columns if c.startswith(prefix+'_')]
  if arm.endswith('_action'):added=[c for c in added if '_size_' not in c]
  if arm.endswith('_size'):added=[c for c in added if '_size_' in c]
  cols=old+added;assert len(added) in [32,128,160];X=d.select(cols).to_numpy();assert np.isfinite(X).all();pred=np.zeros((len(d),2));audit=[];(root/'columns.json').write_text(json.dumps(cols,indent=2))
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    a,b,ea,eb,va=targets(d,f,fam)
    for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
     assert not np.any(e&va);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);path=root/f'event{k}_{fam}_fold{f}.cbm'
     if path.exists():m.load_model(str(path))
     else:m.fit(X[e],y[e].astype(int));m.save_model(str(path))
     pred[va,k-1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'family':fam,'head':k,'training_rows':int(e.sum()),'positive_rows':int(y[e].sum()),'validation_rows':int(va.sum()),'training_validation_overlap':int((e&va).sum())})
    print(arm,f,fam,round(time.time()-start,1),flush=True)
  d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(root/'event_oof.parquet');(root/'training_audit.json').write_text(json.dumps(audit,indent=2));assemble(root)
if __name__=='__main__':main()
