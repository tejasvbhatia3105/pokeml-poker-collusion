import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session15_responses');C=pl.col
def main():
 d=hand_data();extra=pl.read_parquet(ROOT/'hand_features.parquet');players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');lookup={tuple(r[:2]):r[2:] for r in extra.select('pair_id','hand_id',*[f'response_{ctx}_{k}_count' for ctx in ['both_alive','facing_pair','both_raised','facing_lower','facing_higher'] for k in range(4)]).iter_rows()};error=0.;checked=0
 for table in sorted(d['table_id'].unique())[:10]:
  a=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').sort('hand_id','action_no','player_id');actions={h:g for (h,),g in a.group_by('hand_id')};s=pl.read_parquet(f'artifacts/policy/states/{table}.parquet').select('hand_id','player_id','street_no','fold_no');folds={(h,p,st):fo for h,p,st,fo in s.iter_rows()};seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet');net={(h,p):v for h,p,v in seats.select('hand_id','player_id','net_chips').iter_rows()};q=d.filter(C('table_id')==table).select('pair_id','hand_id').join(players,on='pair_id')
  for pid,h,p1,p2 in q.iter_rows():
   counts=np.zeros((5,4));raised=set();last_street=None
   for who,st,no,k,last in actions[h].select('player_id','street_no','action_no','action_class','last_aggressor').iter_rows():
    if st!=last_street:raised=set();last_street=st
    if who not in [p1,p2] and folds[h,p1,int(st)]>=no and folds[h,p2,int(st)]>=no:
     facing=last in [p1,p2];masks=[True,facing,p1 in raised and p2 in raised,(last==p1 and net[h,p1]<=net[h,p2]) or (last==p2 and net[h,p2]<=net[h,p1]),(last==p1 and net[h,p1]>=net[h,p2]) or (last==p2 and net[h,p2]>=net[h,p1])]
     for j,b in enumerate(masks):
      if b:counts[j,k]+=1
    if k==3:raised.add(who)
   error=max(error,float(abs(counts.ravel()-lookup[pid,h]).max()));checked+=1
 assert error==0;report={'chronological_pair_hands_checked':checked,'independent_context_action_count_error':error,'endpoint_swap':json.load(open(ROOT/'feature_audit.json'))};z=d.join(extra,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');report['event_replays']={}
 for arm in ['all','plain']:
  root=ROOT/arm;cols=json.load(open(root/'columns.json'));X=z.select(cols).to_numpy();q=z.select('pair_id','hand_id','fold','behavior_family').join(pl.read_parquet(root/'event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    va=((q['fold']==f)&(q['behavior_family']==fam)).to_numpy()
    for k,c in enumerate(['bg_primary','bg_secondary'],1):
     m=CatBoostClassifier();m.load_model(str(root/f'event{k}_{fam}_fold{f}.cbm'));p=m.predict_proba(X[va],thread_count=2)[:,1];err=max(err,float(abs(p-q[c].to_numpy()[va]).max()))
  assert err<1e-12;report['event_replays'][arm]={'heads':24,'max_error':err}
 (ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
