\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session61_error_audit');C=pl.col
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id','conditional_family');route=pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score');d=d.join(r30,on=['pair_id','hand_id'],validate='1:1').join(route,on='pair_id',validate='m:1');rows=[];detail=[]
 for name,path in [('r32','artifacts/evidence_session50_matchup/current'),('pressure59','artifacts/evidence_session59_pressure_equity')]:
  pred=pl.read_parquet(path+'/background_oof.parquet').select('pair_id','hand_id','cat_and_joint');q=d.join(pred,on=['pair_id','hand_id'],validate='1:1').with_columns(pl.when(C('risk_score')<.05).then(C('conditional_family')).otherwise(C('cat_and_joint')).alias('score'))
  for (pid,),g in q.group_by('pair_id'):
   g=g.sort('score','hand_id',descending=[True,False]);y=g['evidence'].to_numpy();den=min(5,int(y.sum()));ap=float((y[:5]*y[:5].cumsum()/np.arange(1,6)).sum()/den);oracle=min(5,float(y[:5].sum()))/den;rows.append({'model':name,'pair_id':pid,'family':g['behavior_family'][0],'table_id':g['table_id'][0],'fold':g['fold'][0],'MAP':ap,'ordering_loss':oracle-ap,'retrieval_loss':1-oracle,**{'oracle_top'+str(k):min(5,float(y[:k].sum()))/den for k in [5,10,12,20]}})
   for rank,r in enumerate(g.select('hand_id','evidence','evidence_rank','subtype').iter_rows(named=True),1):
    if r['evidence']:detail.append({'model':name,'pair_id':pid,'family':g['behavior_family'][0],'predicted_rank':rank,**r})
 pair=pl.DataFrame(rows);hands=pl.DataFrame(detail);pair.write_parquet(ROOT/'pair_errors.parquet');hands.write_parquet(ROOT/'truth_errors.parquet');cols=['MAP','ordering_loss','retrieval_loss','oracle_top5','oracle_top10','oracle_top12','oracle_top20'];report={'purpose':__doc__,'overall':pair.group_by('model').agg(C(cols).mean()).to_dicts(),'families':pair.group_by('model','family').agg(C(cols).mean()).to_dicts(),'true_rank_recall':hands.group_by('model','evidence_rank').agg(pl.len(),(C('predicted_rank')<=5).mean().alias('recall5'),(C('predicted_rank')<=12).mean().alias('recall12')).sort('model','evidence_rank').to_dicts()};(ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
