import json,time
import numpy as np,polars as pl
from session63_pressure_inference import ROOT,load_models,design,pressure_events,C
from session57_isolation_pressure import data
def main():
 ROOT.mkdir(exist_ok=True);_,d,a,ac=data();a=a.with_row_index('cached_action_row');cfg=json.load(open('artifacts/evidence_session59_pressure_equity/config.json'));ex=np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x'];expected_x=np.column_stack([a.select(ac).to_numpy(),d.select(cfg['hand_columns']).to_numpy()[a['row'].to_numpy()],ex]);people=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');models=load_models();reference=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');actions=hands=0;feature_error=prediction_error=0;start=time.time()
 for i,((table,),q) in enumerate(d.group_by('table_id')):
  f=int(q['fold'][0]);q=q.sort('pair_id','time','hand_id').drop('row').with_row_index('row');newa,x,extra=design(q,people,cfg);keys=['pair_id','hand_id','street_no','action_no'];lookup=newa.select(keys).join(a.select(*keys,'cached_action_row'),on=keys,validate='1:1',maintain_order='left');assert lookup['cached_action_row'].null_count()==0 and len(newa)==len(a.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi'));expected=expected_x[lookup['cached_action_row'].to_numpy()];feature_error=max(feature_error,float(abs(x-expected).max(initial=0)));g=newa['row'].to_numpy();from session27_donor_call_witness import noisy_or
  z=q.select('pair_id','hand_id').join(reference,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
  for k,col in enumerate(['bg_primary','bg_secondary']):
   p=models['pressure'][f][k].predict_proba(x,thread_count=2)[:,1] if len(x) else np.empty(0);pred=noisy_or(p,g,len(q));prediction_error=max(prediction_error,float(abs(pred-z[col].to_numpy()).max()))
  actions+=len(newa);hands+=len(q)
  if i%25==0:print('pressure inference verification',i,actions,hands,round(time.time()-start,1),flush=True)
 assert feature_error==0 and prediction_error==0;out={'all_public_isolation_hands':hands,'raw_pressure_actions':actions,'training_design_feature_error':feature_error,'heldout_event_probability_error':prediction_error,'explicit_development_endpoints':True};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
