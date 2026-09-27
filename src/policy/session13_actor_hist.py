import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data,targets
from session12_hist_replacement import assemble
ROOT=Path('artifacts/evidence_session13_actor/global_hist')
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data().join(pl.read_parquet(ROOT.parent/'hand_features.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');cols=json.load(open(ROOT.parent/'global/columns.json'));X=d.select(cols).to_numpy();pred=np.zeros((len(d),2));audit=[];start=time.time();(ROOT/'columns.json').write_text(json.dumps(cols,indent=2))
 with threadpool_limits(limits=2):
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    a,b,ea,eb,va=targets(d,f,fam)
    for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
     assert not np.any(e&va);path=ROOT/f'event{k}_{fam}_fold{f}.joblib'
     if path.exists():m=joblib.load(path)
     else:m=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+f);m.fit(X[e],y[e].astype(int));joblib.dump(m,path,compress=3)
     pred[va,k-1]=m.predict_proba(X[va])[:,1];audit.append({'fold':f,'family':fam,'head':k,'training_rows':int(e.sum()),'training_validation_overlap':int((e&va).sum())})
    print('global hist',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('new_hist_primary',pred[:,0]),pl.Series('new_hist_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT,ROOT.parent/'global/event_oof.parquet')
if __name__=='__main__':main()
