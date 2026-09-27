\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
C=pl.col;ROOT=Path('artifacts/evidence_session40_isolation_actor')
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data().filter(C('behavior_family')=='coordinated_isolation');a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet').filter((C('action_class')==3)&~C('facing_partner')&C('mw_alive')&(C('players_active')>=3));labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');parts=[]
 for (table,),q in d.group_by('table_id'):
  raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').select('hand_id','street_no','action_no','player_id');z=a.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi').join(raw,on=['hand_id','street_no','action_no'],validate='m:1').join(labs,on='pair_id',validate='m:1');assert z.select(((C('player_id')==C('player_1'))|(C('player_id')==C('player_2'))).all()).item();z=z.with_columns((C('player_id')==C('player_2')).cast(pl.Int8).alias('actor'));parts.append(z.group_by('pair_id','hand_id').agg(*[(C('actor')==r).any().alias(f'actor{r}_pressure') for r in [0,1]]))
 support=d.select('pair_id','hand_id').join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1').fill_null(False);support.write_parquet(ROOT/'support.parquet');q=d.join(support,on=['pair_id','hand_id'],validate='1:1');records=[]
 for (pid,),g in q.group_by('pair_id'):
  p=g.filter(C('evidence')==1).sort('evidence_rank');t=p['time'].to_numpy();s=p.select('actor0_pressure','actor1_pressure').to_numpy();n=len(p);options=[]
  for k in range(n+1):
   if not np.all(np.diff(t[:k])>0) or not np.all(np.diff(t[k:])>0):continue
   for actor in range(2):
    if s[:k,actor].all() and s[k:,1-actor].all():options.append({'cut':k,'primary_actor':actor})
  records.append({'pair_id':pid,'options':options,'single_actor_evidence_hands':int((s.sum(1)==1).sum()),'both_actor_evidence_hands':int((s.sum(1)==2).sum())})
 report={'method':__doc__,'pairs':len(records),'feasible_pairs':sum(bool(r['options']) for r in records),'unique_option_pairs':sum(len(r['options'])==1 for r in records),'unique_priority_actor_pairs':sum(len({o['primary_actor'] for o in r['options']})==1 for r in records),'single_actor_evidence_hands':sum(r['single_actor_evidence_hands'] for r in records),'both_actor_evidence_hands':sum(r['both_actor_evidence_hands'] for r in records),'records':records};(ROOT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
if __name__=='__main__':main()
