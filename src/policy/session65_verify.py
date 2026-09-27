import json,time
import numpy as np,polars as pl
from session8_data import hand_data
from session65_list_pressure_inference import ROOT,GROUND_COLUMNS,load_models,events,tree_score,score_pair,C,NAMES
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();original=d.select('pair_id','hand_id','row');px=np.load('artifacts/evidence_session62_grounded_list_boost/grounded_features.npz')['x'];people=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');models=load_models();trees=pl.read_parquet('artifacts/evidence_session62_grounded_list_boost/conditional_boost_oof.parquet').select('pair_id','hand_id','full');equal=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id','equal');ev32=pl.read_parquet('artifacts/evidence_session50_matchup/current/event_oof.parquet');ev59=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');old={f:pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').select('pair_id','hand_id',*[C(n).alias(f'{n}_{f}') for n in NAMES]) for f in range(4)};fe=ee=te=se=0;hands=pairs=0;start=time.time()
 for i,((table,),q) in enumerate(d.group_by('table_id')):
  f=int(q['fold'][0]);q=q.join(old[f],on=['pair_id','hand_id'],validate='1:1');z=events(q,people,models,[f]);ix=z.select('pair_id','hand_id').join(original,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['row'].to_numpy();fe=max(fe,float(abs(z.select(GROUND_COLUMNS).to_numpy()-px[ix]).max()))
  for prefix,reference in [('list',ev32),('new',ev59)]:
   r=z.select('pair_id','hand_id',f'{prefix}_primary_{f}',f'{prefix}_secondary_{f}').join(reference,on=['pair_id','hand_id'],validate='1:1')
   for head in ['primary','secondary']:ee=max(ee,float(abs(r[f'{prefix}_{head}_{f}']-r[f'bg_{head}']).max()))
  for _,g in z.group_by('pair_id'):
   g=g.sort('time','hand_id');want=g.select('pair_id','hand_id').join(trees,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['full'].to_numpy();te=max(te,float(abs(tree_score(g,f,models)-want).max()));want=g.select('pair_id','hand_id').join(equal,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['equal'].to_numpy();se=max(se,float(abs(score_pair(g,f,models)-want).max()));pairs+=1
  hands+=len(z)
  if i%40==0:print('equal raw inference',i,hands,pairs,'errors',fe,ee,te,se,round(time.time()-start,1),flush=True)
 assert fe==0 and ee==0 and te==0 and se==0;out={'public_hands':hands,'public_pairs':pairs,'grounded_raw_feature_error':fe,'R32_and_pressure_event_error':ee,'tree_score_error':te,'equal_score_error':se,'all_four_heldout_folds_replayed':True};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
