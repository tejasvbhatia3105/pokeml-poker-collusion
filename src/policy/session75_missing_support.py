\
\
\
\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session9_pair_baseline import SEQ
from session73_exclusion_audit import metrics,blend
ROOT=Path('artifacts/pair_session75_missing_support');C=pl.col
MODELS=['r4s','r4k',*SEQ];WEIGHTS=np.array([.25,.25,*([1/15]*6),.1])
def controls(q):
 p=q.select([v+'_lpo' for v in MODELS]).to_numpy();orig=q.select([v+'_orig' for v in MODELS]).to_numpy();kept=q.select([v+'_kept' for v in MODELS]).to_numpy();new=q.select([v+'_new' for v in MODELS]).to_numpy();floors=np.array([1e-6,1e-6,*([1e-300]*7)]);lp=np.log(np.maximum(p,floors));baseline=np.exp(lp@WEIGHTS);absent=(p<orig-1e-12)&(kept<10)&(new==0);weights=WEIGHTS[None,:]*(~absent);total=weights.sum(1);abstain=np.exp((weights*lp).sum(1)/np.maximum(total,1e-300));abstain=np.where(total>0,abstain,baseline);g=np.sqrt(np.maximum(p[:,0],1e-6)*np.maximum(p[:,1],1e-6));s6=p[:,2:5].mean(1);s7=p[:,5:8].mean(1);seq=np.exp((np.log(s6.clip(1e-300))+np.log(s7.clip(1e-300))+.5*np.log(p[:,8].clip(1e-300)))/2.5);arithmetic=np.sqrt(g*seq);return baseline,abstain,arithmetic,absent
def main():
 ROOT.mkdir(exist_ok=True);reports=[]
 for w in ['full','first_2000','last_2000']:
  q=pl.read_parquet(f'artifacts/pair_session73_exclusion_audit/{w}.parquet').rename({'risk_score':'archived_r26'})
  for v in ['r4s','r4k']:
   z=pl.read_parquet(f'artifacts/pair_session74_complete_rescore/{v}_{w}.parquet');q=q.drop(*[v+s for s in ['_lpo','_orig','_new','_kept']]).join(z.select('pair_id',(1-C('none')).alias(v+'_lpo'),C('risk_orig').alias(v+'_orig'),C('risk_new').alias(v+'_new'),C('n_kept').alias(v+'_kept')),on='pair_id',validate='1:1')
  baseline,abstain,arithmetic,absent=controls(q);np.testing.assert_allclose(baseline,blend(q,'_lpo'),rtol=1e-12,atol=1e-14);q=q.with_columns(pl.Series('risk_score',baseline),pl.Series('abstain',abstain),pl.Series('arithmetic_seeds',arithmetic),pl.Series('absent_components',absent.sum(1)));q.write_parquet(ROOT/f'{w}.parquet');rows={k:metrics(q,k) for k in ['archived_r26','risk_score','abstain','arithmetic_seeds']};known=q.filter(C('label')>=0);report={'window':w,'metrics':rows,'affected_known_targets':known.filter((C('label')==1)&(C('absent_components')>0)).select('pair_id','n_hands','n_truth','risk_score','abstain','absent_components').to_dicts(),'affected_confirmed_negatives':known.filter((C('label')==0)&(C('absent_components')>0)).height,'unknown_with_abstention':q.filter((C('label')<0)&(C('absent_components')>0)).height,'unknown_crossing_05':q.filter((C('label')<0)&(C('risk_score')<.5)&(C('abstain')>=.5)).height};reports.append(report)
 (ROOT/'report.json').write_text(json.dumps({'method':__doc__,'windows':reports},indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
