import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
C=pl.col;ROOT=Path('artifacts/evidence_session85_r33_error_audit')
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();q=d.join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id','equal'),on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id','conditional_family'),on=['pair_id','hand_id'],validate='1:1').join(pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score'),on='pair_id',validate='m:1').with_columns(pl.when(C('risk_score')<.05).then(C('conditional_family')).otherwise(C('equal')).alias('score'));pairs=[];hands=[]
 for (pid,),g in q.group_by('pair_id'):
  g=g.sort('score','hand_id',descending=[True,False]);y=g['evidence'].to_numpy();den=min(5,int(y.sum()));ap=float((y[:5]*y[:5].cumsum()/np.arange(1,6)).sum()/den);o=min(5,float(y[:5].sum()))/den;meta={'pair_id':pid,'family':g['behavior_family'][0],'table_id':g['table_id'][0],'fold':g['fold'][0]};pairs.append({**meta,'MAP':ap,'ordering_loss':o-ap,'retrieval_loss':1-o,**{f'oracle_top{k}':min(5,float(y[:k].sum()))/den for k in [5,6,8,10,12,20]}})
  for rank,r in enumerate(g.select('hand_id','evidence','evidence_rank','score').iter_rows(named=True),1):
   if r['evidence']:hands.append({**meta,**r,'predicted_rank':rank})
 p=pl.DataFrame(pairs);h=pl.DataFrame(hands);p.write_parquet(ROOT/'pair_errors.parquet');h.write_parquet(ROOT/'truth_ranks.parquet');cols=['MAP','ordering_loss','retrieval_loss']+[f'oracle_top{k}' for k in [5,6,8,10,12,20]];report={'purpose':__doc__,'actual_best_score':.93048,'required_evidence_MAP_gain_if_other_components_fixed':(.94-.93048)/.2,'overall':p.select(C(cols).mean()).to_dicts()[0],'families':p.group_by('family').agg(C(cols).mean()).to_dicts(),'counts':{'pairs':len(p),'truth_hands':len(h),'truth_outside5':int((h['predicted_rank']>5).sum()),'truth_ranks6to10':int(((h['predicted_rank']>5)&(h['predicted_rank']<=10)).sum()),'truth_outside10':int((h['predicted_rank']>10).sum())},'limits':['Oracles use truth only for diagnosis; not deployable predictions.','Local and competition MAP need not match.','These validation pools have been repeatedly reused.']};assert abs(report['overall']['MAP']-.7973334826762246)<1e-12;(ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
