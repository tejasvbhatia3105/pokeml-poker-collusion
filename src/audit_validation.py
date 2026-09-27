import hashlib,json,runpy
from pathlib import Path
import numpy as np,pandas as pd
from train_baseline import ap
before=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()
v=runpy.run_path('src/evaluate_combined.py')
p=v['pair'].copy();p['evidence_ap']=p.pair_id.map(v['es']).fillna(0)
f=pd.read_parquet('artifacts/pair_features.parquet',columns=['pair_id','phase','table_id']);f=f[f.phase=='development']
p=p.merge(f[['pair_id','table_id']],on='pair_id',validate='one_to_one').sort_values('pair_id').reset_index(drop=True)
folds=json.loads(Path('artifacts/folds.json').read_text());table_fold={}
for f in folds:
 for table in f['valid_tables']:
  assert table not in table_fold,'Pool appears in multiple validation folds'
  table_fold[table]=f['fold']
assert set(p.table_id)<=set(table_fold)
p['fold']=p.table_id.map(table_fold)
names=v['names'];y=p.truth.to_numpy();risk=p.risk_score.to_numpy();family=p.family.to_numpy();e=p.evidence_ap.to_numpy()
def score(ix):
 yy=y[ix];rr=risk[ix];ff=family[ix]
 pair_ap=ap(yy>0,rr)
 behavior=np.mean([ap(yy==i,rr*(ff==n)*(rr>=.01)) for i,n in enumerate(names[1:],1)])
 evidence=float(e[ix][yy>0].mean())
 return float(.7*pair_ap+.2*evidence+.1*behavior)
by_fold={str(i):score(np.flatnonzero(p.fold.to_numpy()==i)) for i in range(4)}
blocks=[np.array(ix,dtype=int) for ix in p.groupby('table_id').groups.values()]
rng=np.random.default_rng(20260911);scores=[]
for _ in range(2000):
 ix=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]);ix=ix[np.argsort(p.pair_id.to_numpy()[ix],kind='stable')]
 scores.append(score(ix))
report={'score':score(np.arange(len(p))),'fold_scores':by_fold,'pool_bootstrap_95_percentile_interval':np.quantile(scores,[.025,.975]).tolist(),'bootstrap_replicates':len(scores),'validation_pools':len(blocks),'public_test_score':None,'private_test_score':None,'submission_sha256':before,'limitations':['Bootstrap measures sampling variability of fixed validation predictions, not training variability or fold-based model selection bias.','Unseen coordination family is absent from public positive labels.','Kaggle population prevalence and timeline differ from labelled development validation.']}
assert hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()==before
Path('artifacts/validation_audit.json').write_text(json.dumps(report,indent=2))
p.to_csv('artifacts/combined_oof_audit.csv',index=False)
print(json.dumps(report,indent=2))
