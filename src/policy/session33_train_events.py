import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session33_rollout')
def main():
 d=hand_data();f=pl.read_parquet(ROOT/'hand_features.parquet');naive=[c for c in f.columns if c.startswith('claims_naive_')];side=[c for c in f.columns if c.startswith('claims_side_')];assert len(naive)==len(side)==144;d=d.join(f,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');base=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];start=time.time();audit=[]
 for kind,extra in [('naive',naive),('sidepot',naive+side)]:
  root=ROOT/kind;root.mkdir(exist_ok=True);cols=base+extra;(root/'columns.json').write_text(json.dumps(cols,indent=2));(root/'config.json').write_text(json.dumps({'method':__doc__,'arm':kind,'new_features':len(extra),'model':'original Cat400 depth5 lr .035 L2 8, same seeds and priority-censored targets','source':'version3 action-gated claims features; all_in calls distinguished from raises by amount vs to_call; v2 buggy models archived separately'},indent=2));X=d.select(cols).to_numpy();assert np.isfinite(X).all();pred=np.zeros((len(d),2))
  for fold in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    a,b,ea,eb,va=targets(d,fold,fam)
    for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
     assert not (e&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*fold+k,verbose=False,allow_writing_files=False);m.fit(X[e],y[e]);m.save_model(str(root/f'event{k}_{fam}_fold{fold}.cbm'));pred[va,k-1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'arm':kind,'fold':fold,'family':fam,'head':k,'training_rows':int(e.sum()),'validation_overlap':0})
    print('sidepot event',kind,fold,fam,round(time.time()-start,1),flush=True)
  d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(root/'event_oof.parquet');assemble(root)
 (ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
