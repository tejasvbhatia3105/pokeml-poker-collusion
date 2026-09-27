import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
C=pl.col;ROOT=Path('artifacts/evidence_session28_soft_actor')
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data().filter(C('behavior_family')=='soft_play');a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet').select('pair_id','hand_id','street_no','action_no','action_class','facing_partner','players_active','mw_alive');labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');parts=[]
 for (table,),q in d.group_by('table_id'):
  raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').select('hand_id','street_no','action_no','player_id');z=a.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi').join(raw,on=['hand_id','street_no','action_no'],validate='m:1').join(labs,on='pair_id',validate='m:1');assert z.select(((C('player_id')==C('player_1'))|(C('player_id')==C('player_2'))).all()).item();z=z.with_columns((C('player_id')==C('player_2')).cast(pl.Int8).alias('actor'));parts.append(z.group_by('pair_id','hand_id').agg(*[((C('actor')==r)&(C('action_class')==k)&(C('facing_partner') if k!=1 else ((C('players_active')==2)&C('mw_alive')))).any().alias(f'actor{r}_{name}') for r in range(2) for k,name in [(0,'fold'),(2,'call'),(1,'check')]]))
 sup=pl.concat(parts);sup.write_parquet(ROOT/'support.parquet');q=d.join(sup,on=['pair_id','hand_id'],validate='1:1');records=[]
 for (pid,),g in q.group_by('pair_id'):
  p=g.filter(C('evidence')==1).sort('evidence_rank');t=p['time'].to_numpy();n=len(p);options=[]
  for i in range(n+1):
   for j in range(i,n+1):
    typ=np.r_[np.zeros(i,int),np.ones(j-i,int),np.full(n-j,2)]
    if not all(np.all(np.diff(t[typ==c])>0) for c in range(3)):continue
    s=np.stack([p.select(*[f'actor{r}_{k}' for k in ['fold','call','check']]).to_numpy()[np.arange(n),typ] for r in range(2)],1)
    if s.any(1).all():options.append({'fold_cut':i,'call_cut':j,'consistent_actors':np.flatnonzero(s.all(0)).tolist()})
  actors=sorted({r for o in options for r in o['consistent_actors']});records.append({'pair_id':pid,'options':options,'consistent_actors':actors})
 out={'purpose':__doc__,'pairs':len(records),'grounded_feasible':sum(bool(r['options']) for r in records),'persistent_feasible':sum(bool(r['consistent_actors']) for r in records),'unique_actor':sum(len(r['consistent_actors'])==1 for r in records),'records':records};(ROOT/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
if __name__=='__main__':main()
