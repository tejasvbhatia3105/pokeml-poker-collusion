\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl,joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data,targets
from session12_event_replacement import assemble
from session12_hist_replacement import assemble as hist_assemble
C=pl.col;ROOT=Path('artifacts/evidence_session31_conditional_events')
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'models':'unchanged Cat400 depth5 and HGB300 leaf15 schedules','features':'unchanged811','supervision':'eligible_secondary & eligible_primary & ~primary_target; original targets only outer-training labels','inference':'primary original; secondary (1-primary)*new conditional posterior','caveat':'head targets remain inferred/censored labels; this changes factorization, not public truth'},indent=2));d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();raw=pl.concat([pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).select('pair_id','hand_id','cat_primary','hist_primary') for f in range(4)]);q=d.select('pair_id','hand_id').join(raw,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');ca=q['cat_primary'].to_numpy();ha=q['hist_primary'].to_numpy();cs=np.zeros(len(d));hs=np.zeros(len(d));audit=[];start=time.time()
 with threadpool_limits(limits=2):
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    a,b,ea,eb,va=targets(d,f,fam);e=eb&ea&~a;assert not (e&va).any();assert (b&e).sum()==(b&eb).sum();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6312+11*f,verbose=False,allow_writing_files=False);m.fit(X[e],b[e]);m.save_model(str(ROOT/f'secondary_{fam}_fold{f}.cbm'));cs[va]=m.predict_proba(X[va],thread_count=2)[:,1];h=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+f);h.fit(X[e],b[e]);joblib.dump(h,ROOT/f'secondary_{fam}_fold{f}.joblib',compress=3);hs[va]=h.predict_proba(X[va])[:,1];audit.append({'fold':f,'family':fam,'original_eligible_secondary':int(eb.sum()),'conditional_eligible_secondary':int(e.sum()),'positive_secondary':int(b[e].sum()),'validation_overlap':0});print('conditional events',f,fam,round(time.time()-start,1),flush=True)
 out=d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',ca),pl.Series('bg_secondary',(1-ca)*cs),pl.Series('new_hist_primary',ha),pl.Series('new_hist_secondary',(1-ha)*hs),pl.Series('cat_conditional_secondary',cs),pl.Series('hist_conditional_secondary',hs));out.write_parquet(ROOT/'event_oof.parquet');assert np.max(ca+(1-ca)*cs)<=1+1e-12;assert np.max(ha+(1-ha)*hs)<=1+1e-12;(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT);hist_assemble(root=ROOT,catpath=ROOT/'event_oof.parquet')
if __name__=='__main__':main()
