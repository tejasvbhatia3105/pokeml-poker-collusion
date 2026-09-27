import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data,targets
ROOT=Path('artifacts/evidence_session12/hist_pseudo');C=pl.col
CONFIG={'max_iter':300,'learning_rate':.05,'max_leaf_nodes':15,'min_samples_leaf':20,'l2_regularization':10,'max_bins':127,'early_stopping':False,'pseudo_weight_mass_fraction':.5,'pseudo':'same strict outer-fold pair-gated soft examples as Cat student; weighted class duplication','seed':'4710+fold; unchanged HGB schedule'}
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();pred=np.zeros((len(d),2));audit=[];start=time.time()
 with threadpool_limits(limits=2):
  for f in range(4):
   q=pl.read_parquet(list(Path(f'artifacts/evidence_session12/unlabelled_pseudo/outer{f}').glob('T*.parquet'))).join(pl.read_parquet(f'artifacts/evidence_session12/pseudo_pair_gate/gate_fold{f}.parquet').filter(C('keep')).select('pair_id'),on='pair_id',validate='m:1');assert (q['pool_fold']!=f).all() and (q['teacher_fold']==f).all()
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    z=q.filter(C('behavior_family')==fam);ZX=z.select(cols).to_numpy();a,b,ea,eb,va=targets(d,f,fam)
    for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
     col='primary' if k==1 else 'secondary';use=z['eligible_'+col].to_numpy();px=ZX[use];pr=z['pseudo_'+col].to_numpy()[use];ix=np.flatnonzero(e);assert not np.any(e&va);weight=.5*len(ix)/max(1,len(px));tx=np.concatenate([X[ix],px,px]);ty=np.r_[y[ix].astype(int),np.zeros(len(px),int),np.ones(len(px),int)];w=np.r_[np.ones(len(ix)),weight*(1-pr),weight*pr];path=ROOT/f'event{k}_{fam}_fold{f}.joblib'
     if path.exists():m=joblib.load(path)
     else:m=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+f);m.fit(tx,ty,sample_weight=w);joblib.dump(m,path,compress=3)
     pred[va,k-1]=m.predict_proba(X[va])[:,1];audit.append({'fold':f,'family':fam,'kind':k,'original_rows':len(ix),'pseudo_rows':len(px),'pseudo_positive_mass':float(pr.sum())})
    print('hist pseudo',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('new_hist_primary',pred[:,0]),pl.Series('new_hist_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2))
 from session12_hist_replacement import assemble
 assemble()
if __name__=='__main__':main()
