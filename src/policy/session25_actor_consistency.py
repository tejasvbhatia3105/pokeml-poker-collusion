\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
C=pl.col
ROOT=Path('artifacts/evidence_session25_actor_consistency')

def main():
 ROOT.mkdir(exist_ok=True)
 d=hand_data();labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
 a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet').select('pair_id','hand_id','street_no','action_no','action_class','facing_partner')
 parts=[];start=time.time()
 for (table,),q in d.group_by('table_id'):
  z=a.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi')
  raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').select('hand_id','street_no','action_no','player_id')
  z=z.join(raw,on=['hand_id','street_no','action_no'],validate='m:1').join(labs,on='pair_id',validate='m:1')
  assert z.select(((C('player_id')==C('player_1'))|(C('player_id')==C('player_2'))).all()).item()
  z=z.with_columns((C('player_id')==C('player_2')).cast(pl.Int8).alias('actor'))
  parts.append(z.group_by('pair_id','hand_id').agg(*[
   ((C('actor')==r)&C('facing_partner')&(C('action_class')==k)).any().alias(f'actor{r}_{name}')
   for r in range(2) for k,name in [(0,'fold'),(2,'call')]]))
 support=pl.concat(parts);assert len(support)==len(d);support.write_parquet(ROOT/'support.parquet')
 q=d.select('pair_id','hand_id','behavior_family','time','evidence','evidence_rank').join(support,on=['pair_id','hand_id'],validate='1:1')
 records=[]
 for (pid,),g in q.filter(C('behavior_family')=='directed_transfer').group_by('pair_id'):
  p=g.filter(C('evidence')==1).sort('evidence_rank');t=p['time'].to_numpy();n=len(p);options=[]
  for k in range(n+1):
   typ=np.r_[np.zeros(k,int),np.ones(n-k,int)]
   if not all(np.all(np.diff(t[typ==c])>0) for c in range(2)):continue
   s=np.stack([p.select(f'actor{r}_fold',f'actor{r}_call').to_numpy()[np.arange(n),typ] for r in range(2)],1)
   if not s.any(1).all():continue
   options.append({'cut':k,'consistent_actors':np.flatnonzero(s.all(0)).tolist(),'unique_actor_hands':int((s.sum(1)==1).sum()),'actor_support':s.astype(int).tolist()})
  records.append({'pair_id':pid,'truth_count':n,'options':options,'grounded_feasible':bool(options),'persistent_feasible':any(o['consistent_actors'] for o in options)})
 report={'purpose':__doc__,'pairs':len(records),'grounded_feasible':sum(r['grounded_feasible'] for r in records),'persistent_feasible':sum(r['persistent_feasible'] for r in records),'records':records,'seconds':time.time()-start}
 (ROOT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
if __name__=='__main__':main()
