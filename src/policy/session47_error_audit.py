import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import polars as pl,numpy as np
from session8_data import hand_data
C=pl.col;ROOT=Path('artifacts/evidence_session47_error_audit')
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();p=pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/background_oof.parquet');r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id','conditional_family');route=pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score');d=d.join(p,on=['pair_id','hand_id'],validate='1:1').join(r30,on=['pair_id','hand_id'],validate='1:1').join(route,on='pair_id',validate='m:1').with_columns(pl.when(C('risk_score')<.05).then(C('conditional_family')).otherwise(C('cat_and_joint')).alias('s'));rows=[];detail=[]
 for (pid,),g in d.group_by('pair_id'):
  g=g.sort('s','hand_id',descending=[True,False]);y=g['evidence'].to_numpy();den=min(5,y.sum());rows.append({'pair_id':pid,'family':g['behavior_family'][0],'table_id':g['table_id'][0],'fold':g['fold'][0],'map5':float((y[:5]*y[:5].cumsum()/np.arange(1,6)).sum()/den),**{'oracle_from_top'+str(k):min(5,float(y[:k].sum()))/den for k in [5,10,20,40]}})
  for i,r in enumerate(g.select('hand_id','evidence','evidence_rank','subtype','fold_partner','call_partner','agg_partner','check_hu').iter_rows(named=True)):
   if r['evidence']==1:detail.append({'pair_id':pid,'family':g['behavior_family'][0],'predicted_rank':i+1,**r})
 r=pl.DataFrame(rows);h=pl.DataFrame(detail);r.write_parquet(ROOT/'pair_errors.parquet');h.write_parquet(ROOT/'true_hand_errors.parquet');cols=['map5']+['oracle_from_top'+str(k) for k in [5,10,20,40]];report={'purpose':__doc__,'overall':r.select(C(cols).mean()).to_dicts(),'families':r.group_by('family').agg(C(cols).mean()).to_dicts(),'by_evidence_rank':h.group_by('evidence_rank').agg(pl.len(),(C('predicted_rank')<=5).mean().alias('recall5'),(C('predicted_rank')<=10).mean().alias('recall10'),C('predicted_rank').median()).sort('evidence_rank').to_dicts(),'missed_beyond20':h.filter(C('predicted_rank')>20).height};(ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
