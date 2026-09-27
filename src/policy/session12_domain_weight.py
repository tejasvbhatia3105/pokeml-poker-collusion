\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
ROOT=Path('artifacts/evidence_session12/domain_weight_normalized');C=pl.col
CONFIG={'domain_iterations':150,'domain_depth':3,'domain_learning_rate':.05,'domain_l2':10,'domain_class_weight':'balanced, mean-one training weight','ratio_clip':[.25,4.],'domain_crossfit':'outer and inner pools excluded; three remaining original folds used for inner crossfit','event':'unchanged 400-tree Cat head, mean-one density ratio over original eligible rows; no pseudo targets'}
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];q=pl.read_parquet('artifacts/evidence_session12/future_pseudo/future_features.parquet');X=d.select(cols).to_numpy();Q=q.select(cols).to_numpy();fv=d['fold'].to_numpy();qf=q['fold'].to_numpy();fams=['directed_transfer','soft_play','coordinated_isolation'];family=d['behavior_family'].to_numpy();qfamily=q['behavior_family'].to_numpy();DX=np.column_stack([X,*[(family==b).astype(float) for b in fams]]);DQ=np.column_stack([Q,*[(qfamily==b).astype(float) for b in fams]]);pred=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  weights=np.full(len(d),np.nan)
  for inner in range(4):
   if inner==f:continue
   tr=(fv!=f)&(fv!=inner);qt=(qf!=f)&(qf!=inner);held=fv==inner;m=CatBoostClassifier(iterations=150,depth=3,learning_rate=.05,l2_leaf_reg=10,thread_count=2,random_seed=12300+10*f+inner,verbose=False,allow_writing_files=False);path=ROOT/f'domain_outer{f}_inner{inner}.cbm'
   if path.exists():m.load_model(str(path))
   else:m.fit(np.r_[DX[tr],DQ[qt]],np.r_[np.zeros(tr.sum()),np.ones(qt.sum())],sample_weight=np.r_[np.full(tr.sum(),.5*(tr.sum()+qt.sum())/tr.sum()),np.full(qt.sum(),.5*(tr.sum()+qt.sum())/qt.sum())]);m.save_model(str(path))
   p=m.predict_proba(DX[held],thread_count=2)[:,1];weights[held]=np.clip(p/(1-p),.25,4);assert not np.any(held&tr)
  assert np.isnan(weights[fv==f]).all();assert np.isfinite(weights[fv!=f]).all();np.save(ROOT/f'weights_fold{f}.npy',weights)
  for fam in fams:
   a,b,ea,eb,va=targets(d,f,fam)
   for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
    assert not np.any(e&va);w=weights[e];w=w/w.mean();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);path=ROOT/f'event{k}_{fam}_fold{f}.cbm'
    if path.exists():m.load_model(str(path))
    else:m.fit(X[e],y[e],sample_weight=w);m.save_model(str(path))
    pred[va,k-1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'family':fam,'kind':k,'eligible_hands':int(e.sum()),'positive_events':int(y[e].sum()),'weight_quantiles':np.quantile(w,[0,.1,.5,.9,1]).tolist(),'event_mean_weight':float(w[y[e].astype(bool)].mean()),'nonevent_mean_weight':float(w[~y[e].astype(bool)].mean())})
   print('domain weighted event',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2));from session12_event_replacement import assemble;assemble(ROOT)
if __name__=='__main__':main()
