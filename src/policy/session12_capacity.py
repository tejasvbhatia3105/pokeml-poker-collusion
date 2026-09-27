import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
MODE=os.environ.get('CAPACITY_PSEUDO','0')=='1';ROOT=Path('artifacts/evidence_session12/capacity_pseudo' if MODE else 'artifacts/evidence_session12/capacity_control');C=pl.col
CONFIG={'iterations':800,'depth':6,'learning_rate':.03,'l2_leaf_reg':12,'pseudo':MODE,'pair_gate':MODE,'pseudo_mass_fraction':.5,'selection':'one fixed larger architecture; positive-only matched control; no held-out early stop','seed':'same as original event heads'}
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();pred=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  if MODE:
   q=pl.read_parquet(list(Path(f'artifacts/evidence_session12/unlabelled_pseudo/outer{f}').glob('T*.parquet')));assert len(list(Path(f'artifacts/evidence_session12/unlabelled_pseudo/outer{f}').glob('T*.json')))==sum(v!=f for v in json.load(open('artifacts/policy/table_folds.json')).values());assert (q['pool_fold']!=f).all() and (q['teacher_fold']==f).all();q=q.join(pl.read_parquet(f'artifacts/evidence_session12/pseudo_pair_gate/gate_fold{f}.parquet').filter(C('keep')).select('pair_id'),on='pair_id',validate='m:1')
  for fam in ['directed_transfer','soft_play','coordinated_isolation']:
   a,b,ea,eb,va=targets(d,f,fam)
   for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
    assert not np.any(e&va);tx=X[e];ty=y[e].astype(int);w=np.ones(e.sum());extra=0;soft=0.
    if MODE:
     z=q.filter(C('behavior_family')==fam);col='primary' if k==1 else 'secondary';use=z['eligible_'+col].to_numpy();px=z.select(cols).to_numpy()[use];pr=z['pseudo_'+col].to_numpy()[use];weight=.5*e.sum()/max(1,len(px));tx=np.concatenate([tx,px,px]);ty=np.r_[ty,np.zeros(len(px),int),np.ones(len(px),int)];w=np.r_[w,weight*(1-pr),weight*pr];extra=len(px);soft=float(pr.sum())
    m=CatBoostClassifier(iterations=800,depth=6,learning_rate=.03,l2_leaf_reg=12,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);path=ROOT/f'event{k}_{fam}_fold{f}.cbm'
    if path.exists():m.load_model(str(path))
    else:m.fit(tx,ty,sample_weight=w);m.save_model(str(path))
    pred[va,k-1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'family':fam,'event':k,'original_rows':int(e.sum()),'original_positives':int(y[e].sum()),'pseudo_rows':extra,'pseudo_soft_positive_mass':soft})
   print('capacity',MODE,f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2));from session12_event_replacement import assemble;assemble(ROOT)
if __name__=='__main__':main()
