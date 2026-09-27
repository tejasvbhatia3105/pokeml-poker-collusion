\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score
from session9_pair_baseline import assemble,SEQ
ROOT=Path('artifacts/pair_session73_exclusion_audit');C=pl.col
def blend(q,suffix):
 g=np.sqrt(np.clip(q['r4s'+suffix].to_numpy(),1e-6,1)*np.clip(q['r4k'+suffix].to_numpy(),1e-6,1));x=np.log(np.clip(q.select([v+suffix for v in SEQ]).to_numpy(),1e-300,1));seq=np.exp((x[:,:3].mean(1)+x[:,3:6].mean(1)+.5*x[:,6])/2.5);return np.sqrt(g*seq)
def metrics(q,col):
 score=q[col].to_numpy();lab=q['label'].to_numpy();known=lab>=0;y=lab[known];s=score[known];others=np.sort(score[lab!=1]);eligible=(lab==1)&(q['n_truth'].to_numpy()>0);above=len(others)-np.searchsorted(others,score[eligible],side='right');return {'known_AP':average_precision_score(y,s),'negative_weight50_AP':average_precision_score(y,s,sample_weight=np.where(y==0,50,1)),'known_negatives_above_05':int(((lab==0)&(score>=.5)).sum()),'eligible_positive_recall300_other_pairs':float((above<300).mean()),'eligible_positive_recall1000_other_pairs':float((above<1000).mean()),'eligible_positive_below_005':int((eligible&(score<.05)).sum())}
def main():
 ROOT.mkdir(exist_ok=True);reports=[];allrows=[]
 for w in ['full','first_2000','last_2000']:
  q=pl.read_parquet(f'artifacts/evidence_session9/pair_{w}.parquet')
  for v in ['r4s','r4k',*SEQ]:
   path=f'artifacts/evidence_session9/{v}_{w}.csv' if v.startswith('r4') else f'artifacts/{v}/dev_{w}_lpo.csv';z=pl.read_csv(path).select('pair_id',(1-C('none')).alias(v+'_lpo'),C('risk_orig').alias(v+'_orig'),C('risk_new').alias(v+'_new'),C('n_kept').alias(v+'_kept'));q=q.join(z,on='pair_id',validate='1:1')
  q=q.with_columns(pl.Series('replayed',blend(q,'_lpo')),pl.Series('original',blend(q,'_orig')));err=float((q['risk_score']-q['replayed']).abs().max());assert err<1e-12
                                                                            
                                                                        
  affected=[(C(v+'_lpo')<C(v+'_orig')-1e-12) for v in ['r4s','r4k',*SEQ]];thin=[affected[i]&(C(v+'_kept')<10) for i,v in enumerate(['r4s','r4k',*SEQ])];q=q.with_columns(pl.any_horizontal(affected).alias('affected'),pl.any_horizontal(thin).alias('zero_from_under10'))
  rows=[]
  for name,mask in [('known_target',C('label')==1),('eligible_target',(C('label')==1)&(C('n_truth')>0)),('confirmed_negative',C('label')==0),('unknown',C('label')<0)]:
   z=q.filter(mask);rows.append({'group':name,'pairs':len(z),'affected':int(z['affected'].sum()),'zero_from_under10':int(z['zero_from_under10'].sum()),'orig_above_05_lpo_below_005':z.filter((C('original')>=.5)&(C('risk_score')<.05)).height})
  reports.append({'window':w,'replay_error':err,'groups':rows,'metrics':{k:metrics(q,k) for k in ['risk_score','original']}});q.with_columns(pl.lit(w).alias('window')).write_parquet(ROOT/f'{w}.parquet');allrows.append(q.filter((C('label')==1)&(C('affected'))).select('pair_id','true_family','n_hands','n_truth','risk_score','original','zero_from_under10',*[v+'_kept' for v in ['r4s','r4k',*SEQ]]).with_columns(pl.lit(w).alias('window')))
 pl.concat(allrows).write_csv(ROOT/'affected_known_targets.csv');(ROOT/'report.json').write_text(json.dumps({'method':__doc__,'windows':reports},indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
