import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
from session34_river_engine import initialize,action_class,ROOT
C=pl.col;COLS=json.load(open('artifacts/policy/feature_columns.json'));INDEX={c:i for i,c in enumerate(COLS)}
def vector(st,j,template,bb):
 x=template.copy();call=st.to_call(j);stack=st.remaining[j];pot=st.contribution.sum();values={'street_no':3,'action_no':st.action_no,'players_active':st.alive.sum(),'previous_action':st.previous_action,'prior_raises':st.prior_raises,'big_blind':bb,'pot_bb':pot/bb,'call_bb':call/bb,'stack_bb':stack/bb,'pot_odds':call/max(pot+call,1),'call_stack':call/max(stack,1),'position':(j-st.button)%6,'self_last_aggressor':int(st.last_aggressor==j)}
 for name,value in values.items():x[INDEX[name]]=value
 return x.astype(np.float32)
def table_data(table,needed):
 a=pl.read_parquet(f'artifacts/compact/actions/table_id={table}/*.parquet').join(needed,on='hand_id',how='semi').sort('hand_id','action_no');rivers=a.filter(C('street')=='river').select('hand_id').unique();a=a.join(rivers,on='hand_id',how='semi');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(rivers,on='hand_id',how='semi');s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(rivers,on='hand_id',how='semi');p=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').filter(C('street_no')==3).join(rivers,on='hand_id',how='semi');meta={r['hand_id']:r for r in h.to_dicts()};seats={hid:z.sort('seat_no') for (hid,),z in s.group_by('hand_id')};templates={(hid,who):g.select(COLS).to_numpy()[0].astype(np.float32) for (hid,who),g in p.group_by('hand_id','player_id')};reference={(hid,int(n)):row for (hid,n),row in zip(p.select('hand_id','action_no').iter_rows(),p.select(COLS).to_numpy().astype(np.float32))};return a,meta,seats,templates,reference
def main():
 d=hand_data();errors=np.zeros(len(COLS));count=0;missing=0;start=time.time()
 for (table,),q in d.group_by('table_id'):
  a,meta,seats,templates,reference=table_data(table,q.select('hand_id').unique())
  for (hid,),g in a.group_by('hand_id'):
   rows=g.to_dicts();st,index=initialize(meta[hid],seats[hid],rows);people=seats[hid]['player_id'].to_list()
   for j in np.flatnonzero(st.alive&(st.remaining>0)):missing+=int((hid,people[j]) not in templates)
   for row in rows:
    if row['street']!='river':continue
    j=index[row['player_id']];assert st.next_actor()==j;v=vector(st,j,templates[hid,row['player_id']],meta[hid]['big_blind']);errors=np.maximum(errors,abs(v-reference[hid,row['action_no']]));assert not st.apply(j,action_class(row),row['amount']);count+=1
 assert missing==0;assert errors.max()<1e-5;report={'observed_decisions_replayed':count,'max_policy_feature_error':float(errors.max()),'feature_errors':dict(zip(COLS,errors.tolist())),'missing_future_actor_templates':missing,'static_features':'private-card descriptors and style estimates; original style estimates exclude the entire current player-hand/street','dynamic_features':'all 13 state-dependent policy fields rebuilt from engine','seconds':time.time()-start};(ROOT/'policy_state_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
