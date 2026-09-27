\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session9_pair_baseline import assemble,SEQ,N
ROOT=Path('artifacts/evidence_session91_family_uncertainty');C=pl.col
def augment(q):
 p=q.select(N).to_numpy();p=p/np.maximum(p.sum(1,keepdims=True),1e-300);s=np.sort(p,axis=1);return q.with_columns(pl.Series('family_confidence',s[:,-1]),pl.Series('family_margin',s[:,-1]-s[:,-2]),pl.Series('family_entropy',-(p*np.log(p.clip(1e-15))).sum(1)),*[pl.Series('prob_'+n,p[:,i]) for i,n in enumerate(N)])
def stats(q):
 rows=[]
 for name,lo,hi in [('under001',0,.01),('001_to005',.01,.05),('005_to05',.05,.5),('atleast05',.5,1.00001)]:
  z=q.filter((C('risk_score')>=lo)&(C('risk_score')<hi))
  if not len(z):continue
  r={'band':name,'pairs':len(z),'confidence_quantiles':np.quantile(z['family_confidence'].to_numpy(),[0,.1,.5,.9,1]).tolist(),'below06':int((z['family_confidence']<.6).sum()),'below08':int((z['family_confidence']<.8).sum()),'below09':int((z['family_confidence']<.9).sum())}
  if 'true_family' in z.columns:r['wrong_family']=int((z['family']!=z['true_family']).sum())
  rows.append(r)
 return rows
def main():
 ROOT.mkdir(exist_ok=True);seq={n:pl.read_csv(f'artifacts/{n}/eval_all_lpo.csv') for n in SEQ};ev=augment(assemble(pl.read_csv('artifacts/candidate_r12/eval_r4s_all.csv'),pl.read_csv('artifacts/candidate_r12/eval_r4k_all.csv'),seq));submission=pl.read_csv('artifacts/candidate_r33/submission.csv');ev=ev.join(submission.select('pair_id'),on='pair_id',how='semi');assert len(ev)==len(submission)==112540;check=ev.join(submission.select('pair_id',C('risk_score').alias('expected'),C('predicted_behavior').alias('expected_family')),on='pair_id',validate='1:1');err=float((check['risk_score']-check['expected']).abs().max());assert err<1e-12;assert not len(check.filter((C('risk_score')>=.01)&(C('family')!=C('expected_family'))));ev.write_parquet(ROOT/'evaluation.parquet');report={'method':__doc__,'R33_risk_replay_error':err,'evaluation':stats(ev),'development':{}};labs=pl.read_csv('data/development_labels.csv').select('pair_id','label',C('behavior_family').alias('true_family'));truth=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').filter(C('evidence')==1)
 for w,lo,hi in [('full',0,3000),('first_2000',0,2000),('last_2000',1000,3000)]:
  q=augment(assemble(pl.read_parquet(f'artifacts/pair_session74_complete_rescore/r4s_{w}.parquet'),pl.read_parquet(f'artifacts/pair_session74_complete_rescore/r4k_{w}.parquet'),{n:pl.read_csv(f'artifacts/{n}/dev_{w}_lpo.csv') for n in SEQ}));q=q.join(labs,on='pair_id',how='left',validate='1:1');ti=(C('time')*5000).round();nt=truth.filter((ti>=lo)&(ti<hi)).group_by('pair_id').len().rename({'len':'n_truth'});q=q.join(nt,on='pair_id',how='left',validate='1:1').with_columns(C('n_truth').fill_null(0));base=pl.read_parquet(f'artifacts/pair_session75_missing_support/{w}.parquet').select('pair_id',C('risk_score').alias('expected'));r=q.join(base,on='pair_id',validate='1:1');assert float((r['risk_score']-r['expected']).abs().max())<1e-12;q.write_parquet(ROOT/f'{w}.parquet');pos=q.filter((C('label')==1)&(C('n_truth')>0));report['development'][w]={'retaining_published_truth':len(pos),'positives':stats(pos),'wrong_family_rows':pos.filter(C('family')!=C('true_family')).select('pair_id','true_family','family','risk_score','family_confidence','family_margin','n_truth').to_dicts(),'unknowns':stats(q.filter(C('label').is_null()).drop('true_family'))}
 (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
