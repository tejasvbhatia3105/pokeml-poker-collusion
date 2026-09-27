\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
ROOT=Path('artifacts/evidence_session12/pair_weight');C=pl.col
CONFIG={'weight':'inverse eligible-hand count per pair, normalized to mean one; each included relationship has identical total weight','iterations':400,'depth':5,'learning_rate':.035,'l2_leaf_reg':8,'seed':'same archived Cat event seed','data':'original public training-pool event targets only, no pseudo data or validation-derived weights'}
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];x=d.select(cols).to_numpy();pid=d['pair_id'].to_numpy();pred=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  for fam in ['directed_transfer','soft_play','coordinated_isolation']:
   a,b,ea,eb,va=targets(d,f,fam)
   for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
    assert not np.any(e&va);pairs,inv,count=np.unique(pid[e],return_inverse=True,return_counts=True);w=e.sum()/len(pairs)/count[inv];totals=np.bincount(inv,weights=w);assert abs(w.mean()-1)<1e-12 and np.ptp(totals)<1e-8;path=ROOT/f'event{k}_{fam}_fold{f}.cbm';m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False)
    if path.exists():m.load_model(str(path))
    else:m.fit(x[e],y[e],sample_weight=w);m.save_model(str(path))
    pred[va,k-1]=m.predict_proba(x[va],thread_count=2)[:,1];audit.append({'fold':f,'family':fam,'kind':k,'training_pairs':len(pairs),'eligible_hands':int(e.sum()),'positive_events':int(y[e].sum()),'weight_quantiles':np.quantile(w,[0,.1,.5,.9,1]).tolist(),'per_pair_weight_range':float(np.ptp(totals))})
   print('pair weighted',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2));from session12_event_replacement import assemble;assemble(ROOT)
if __name__=='__main__':main()
