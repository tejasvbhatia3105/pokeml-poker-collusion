\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session13_action_events');C=pl.col
CONFIG={'iterations':300,'depth':5,'learning_rate':.035,'l2_leaf_reg':10,'em_rounds':3,'loss':'native CrossEntropy with exact independent-witness posterior marginals','initialization':'equal expected one witness per positive eligible bag','supervision':'original priority-censored event targets; no unlisted-tail negatives','weights':'one unit per action; complete bag noisy-OR likelihood','features':'cached pair-member actions with actual-card multiway context; absolute time, IDs and labels excluded'}
def aggregate(p,g,n):return -np.expm1(np.bincount(g,weights=np.log1p(-np.clip(p,1e-8,1-1e-8)),minlength=n))
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet');cols=[c for c in a.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family','time']];a=a.select('pair_id','hand_id',*cols).join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='m:1',maintain_order='left');X=a.select(cols).to_numpy().astype(np.float32);g=a['row'].to_numpy();assert len(np.unique(g))==len(d);assert np.isfinite(X).all();counts=np.bincount(g,minlength=len(d));pred=np.zeros((len(d),2));audit=[];start=time.time();(ROOT/'columns.json').write_text(json.dumps(cols,indent=2))
 for f in range(4):
  for fam in ['directed_transfer','soft_play','coordinated_isolation']:
   y1,y2,e1,e2,va=targets(d,f,fam)
   for k,(y,e) in enumerate([(y1,e1),(y2,e2)],1):
    tr=np.flatnonzero(e[g]);vi=np.flatnonzero(va[g]);gt=g[tr];assert not np.any(va[gt]);yt=y[gt].astype(float);resp=yt/counts[gt];trace=[]
    for step in range(3):
     m=CatBoostClassifier(iterations=300,depth=5,learning_rate=.035,l2_leaf_reg=10,loss_function='CrossEntropy',thread_count=2,random_seed=13130+11*f+k,verbose=False,allow_writing_files=False);path=ROOT/f'action{k}_{fam}_fold{f}_em{step}.cbm'
     if path.exists():m.load_model(str(path))
     else:m.fit(X[tr],resp);m.save_model(str(path))
     p=m.predict_proba(X[tr],thread_count=2)[:,1];hp=aggregate(p,gt,len(d));resp=np.where(yt>0,p/np.maximum(hp[gt],1e-8),0).clip(0,1);v=np.clip(hp[e],1e-8,1-1e-8);loss=float(-(y[e]*np.log(v)+(1-y[e])*np.log1p(-v)).mean());trace.append(loss)
    p=m.predict_proba(X[vi],thread_count=2)[:,1];pred[va,k-1]=aggregate(p,g[vi],len(d))[va];audit.append({'fold':f,'family':fam,'head':k,'eligible_hands':int(e.sum()),'positive_hands':int(y[e].sum()),'training_actions':len(tr),'heldout_action_overlap':int(va[gt].sum()),'training_bag_nll':trace})
   print('action event',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':main()
