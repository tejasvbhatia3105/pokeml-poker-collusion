\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
ROOT=Path('artifacts/evidence_session12/pseudo_pair_gate');C=pl.col
CONFIG={'iterations':600,'depth':5,'learning_rate':.04,'l2_leaf_reg':10,'seed':'991+fold','training':'confirmed labels only; whole outer pool fold excluded; no early stopping, pseudo targets or validation selection','gate':'pair risk >= .9 and family agrees with event teacher'}
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));labs=pl.read_csv('data/development_labels.csv').select('pair_id','behavior_family');cols=json.load(open('artifacts/policy/residual_columns.json'));folds=json.load(open('artifacts/policy/table_folds.json'));parts=[];names=['none','directed_transfer','soft_play','coordinated_isolation'];start=time.time()
 for path in sorted(Path('artifacts/policy/pair_features').glob('T*.parquet')):
  z=pl.read_parquet(path).filter(C('phase')=='development').join(labs,on='pair_id');parts.append(z.with_columns(pl.lit(folds[path.stem]).alias('fold')))
 d=pl.concat(parts).sort('pair_id');assert len(d)==len(labs);x=d.select(cols).to_numpy();y=np.array([names.index(b) for b in d['behavior_family']]);fv=d['fold'].to_numpy();audit=[]
 for f in range(4):
  tr=fv!=f;m=CatBoostClassifier(iterations=600,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=2,random_seed=991+f,verbose=False,allow_writing_files=False);path=ROOT/f'pair_fold{f}.cbm'
  if path.exists():m.load_model(str(path))
  else:m.fit(x[tr],y[tr]);m.save_model(str(path))
  scores=[]
  for pp in sorted(Path(f'artifacts/evidence_session12/unlabelled_pseudo/outer{f}').glob('T*.parquet')):
   meta=pl.read_parquet(pp,columns=['pair_id','behavior_family','pool_fold','teacher_fold']).unique();assert (meta['pool_fold']!=f).all() and (meta['teacher_fold']==f).all()
   if not len(meta):continue
   z=pl.read_parquet(Path('artifacts/policy/pair_features')/pp.name).filter(C('phase')=='evaluation').join(meta,on='pair_id',validate='1:1');assert len(z)==len(meta);p=m.predict_proba(z.select(cols).to_numpy(),thread_count=2);family=np.array(names[1:])[p[:,1:].argmax(1)];scores.append(z.select('pair_id','behavior_family','pool_fold').with_columns(pl.Series('pair_risk',1-p[:,0]),pl.Series('pair_family',family)))
  q=pl.concat(scores);q=q.with_columns(((C('pair_risk')>=.9)&(C('pair_family')==C('behavior_family'))).alias('keep'));q.write_parquet(ROOT/f'gate_fold{f}.parquet');audit.append({'fold':f,'labelled_training_pairs':int(tr.sum()),'outer_validation_pairs_excluded':int((~tr).sum()),'candidate_pseudo_pairs':len(q),'retained_pseudo_pairs':q['keep'].sum(),'retained_by_family':q.filter(C('keep')).group_by('behavior_family').len().to_dicts()});print('pair pseudo gate',f,audit[-1],'seconds',round(time.time()-start,1),flush=True)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
