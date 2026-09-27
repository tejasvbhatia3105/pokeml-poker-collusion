import json
import numpy as np,polars as pl
from session46_paired_inference import ROOT,load_models,events,score_pair,NAMES
from session8_data import hand_data
C=pl.col
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');models=load_models();expected=pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/event_oof.parquet');scores=pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/background_oof.parquet');records=[]
 for f in range(4):
  counts=d.filter(C('fold')==f).group_by('table_id').agg(C('behavior_family').n_unique().alias('families'),pl.len().alias('hands')).sort(['families','hands'],descending=True);table=counts['table_id'][0];q=d.filter(C('table_id')==table);raw=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').select('pair_id','hand_id',*[C(n).alias(f'{n}_{f}') for n in NAMES]);q=q.join(raw,on=['pair_id','hand_id'],validate='1:1');out=events(q,players,models,[f]);z=out.join(expected.select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1');err=max(float((z[f'new_primary_{f}']-z['bg_primary']).abs().max()),float((z[f'new_secondary_{f}']-z['bg_secondary']).abs().max()));scoreerr=0.
  for _,g in out.group_by('pair_id'):
   g=g.sort('time','hand_id');p=score_pair(g,f,models);ref=g.select('pair_id','hand_id').join(scores,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['cat_and_joint'].to_numpy();scoreerr=max(scoreerr,float(abs(p-ref).max()))
  print('inference verify',f,table,err,scoreerr,flush=True);assert err<1e-12 and scoreerr<1e-7;records.append({'fold':f,'table':table,'hands':len(q),'families':counts['families'][0],'event_probability_error':err,'selection_score_error':scoreerr})
 report={'sampled_multifamily_table_replays':records,'optimized_fold_builder_check':json.load(open('artifacts/eval_paired_feature_api_checks.json'))};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
