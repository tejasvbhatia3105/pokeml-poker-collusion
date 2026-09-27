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
ROOT=Path('artifacts/evidence_session24_rate_control');C=pl.col
def group_weights(p,original_rows,rate):
 high=p>.9;low=p<.02;assert (high|low).all();mass=.5*original_rows;uniform=np.full(len(p),mass/max(1,len(p)))
 if not high.any() or not low.any():return uniform,{'fallback':'one confidence group absent','high':int(high.sum()),'low':int(low.sum())}
 mh=float(p[high].mean());ml=float(p[low].mean());alpha=float(np.clip((rate-ml)/(mh-ml),0,1));w=np.where(high,mass*alpha/high.sum(),mass*(1-alpha)/low.sum());assert abs(w.sum()-mass)<1e-8;actual=float(w@p/mass)
 if ml<=rate<=mh:assert abs(actual-rate)<1e-12
 return w,{'high':int(high.sum()),'low':int(low.sum()),'high_target_mean':mh,'low_target_mean':ml,'high_weight_fraction':alpha,'original_target_rate':rate,'unweighted_pseudo_rate':float(p.mean()),'weighted_pseudo_rate':actual,'fallback':None}
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'data':'same even+odd gated pseudo cache as session23','selection':'post-result control motivated by pseudo positive-rate dilution; original rates estimated from eligible outer-training labels only','student':'unchanged original Cat event schedule and seed'},indent=2));d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();pred=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  even=pl.read_parquet(list(Path(f'artifacts/evidence_session12/unlabelled_pseudo/outer{f}').glob('T*.parquet'))).join(pl.read_parquet(f'artifacts/evidence_session12/pseudo_pair_gate/gate_fold{f}.parquet').filter(C('keep')).select('pair_id'),on='pair_id',validate='m:1');odd=pl.read_parquet(f'artifacts/evidence_session23_more_pseudo/odd_outer{f}.parquet');q=pl.concat([even,odd],how='vertical_relaxed');assert (q['pool_fold']!=f).all() and (q['teacher_fold']==f).all();assert q.select('pair_id','hand_id').n_unique()==len(q)
  for fam in ['directed_transfer','soft_play','coordinated_isolation']:
   z=q.filter(C('behavior_family')==fam);ZX=z.select(cols).to_numpy();a,b,ea,eb,va=targets(d,f,fam)
   for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
    col='primary' if k==1 else 'secondary';use=z['eligible_'+col].to_numpy();px=ZX[use];p=z['pseudo_'+col].to_numpy()[use];ix=np.flatnonzero(e);assert not np.any(e&va);w,info=group_weights(p,len(ix),float(y[ix].mean()));tx=np.concatenate([X[ix],px,px]);ty=np.r_[y[ix].astype(int),np.zeros(len(px),int),np.ones(len(px),int)];weights=np.r_[np.ones(len(ix)),w*(1-p),w*p];m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);path=ROOT/f'event{k}_{fam}_fold{f}.cbm'
    if path.exists():m.load_model(str(path))
    else:m.fit(tx,ty,sample_weight=weights);m.save_model(str(path))
    pred[va,k-1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'family':fam,'head':k,'original_rows':len(ix),'pseudo_rows':len(px),'validation_overlap':int((e&va).sum()),**info})
   print('rate control',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':main()
