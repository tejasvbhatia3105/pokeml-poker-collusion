import json
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
from session63_pressure_inference import ROOT,load_models,events,C
from session46_paired_inference import NAMES,score_pair
def main():
 d=hand_data();players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');models=load_models();eventref=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');scoreref=pl.read_parquet('artifacts/evidence_session59_pressure_equity/background_oof.parquet').select('pair_id','hand_id','cat_and_joint');chosen=[];event_error=score_error=0;hands=0;families=set()
 for f in range(4):
  counts=d.filter(C('fold')==f).group_by('table_id').agg(C('behavior_family').n_unique().alias('families'),pl.len().alias('hands')).sort(['families','hands','table_id'],descending=[True,True,False]);table=counts['table_id'][0];q=d.filter(C('table_id')==table);old=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').select('pair_id','hand_id',*[C(n).alias(f'{n}_{f}') for n in NAMES]);q=q.join(old,on=['pair_id','hand_id'],validate='1:1');z=events(q,players,models,[f]);r=z.select('pair_id','hand_id',f'new_primary_{f}',f'new_secondary_{f}').join(eventref,on=['pair_id','hand_id'],validate='1:1')
  for new,ref in [(f'new_primary_{f}','bg_primary'),(f'new_secondary_{f}','bg_secondary')]:event_error=max(event_error,float(abs(r[new]-r[ref]).max()))
  for _,g in z.group_by('pair_id'):
   g=g.sort('time','hand_id');score=score_pair(g,f,models);ref=g.select('pair_id','hand_id').join(scoreref,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['cat_and_joint'].to_numpy();score_error=max(score_error,float(abs(score-ref).max()))
  chosen.append(table);hands+=len(z);families.update(z['behavior_family'].to_list())
 assert event_error==0 and score_error==0 and len(families)==3;out={'tables':chosen,'folds':4,'hands':hands,'families':sorted(families),'event_replay_error':event_error,'final_score_replay_error':score_error};(ROOT/'end_to_end_verification.json').write_text(json.dumps(out,indent=2));proof=json.load(open(ROOT/'verification.json'));proof['end_to_end']=out;(ROOT/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
