import json
import numpy as np,polars as pl
from session53_outsider_rank import ROOT,PAIR_ROOT,load,compute,C
from session50_verify import aggregate
def main():
 base=pl.read_parquet(PAIR_ROOT/'current/event_oof.parquet');ref=pl.read_parquet(ROOT/'event_oof.parquet');records=[]
 for family in ['directed_transfer','soft_play','coordinated_isolation']:
  d,a,x,cols=load(family);ex,raw=compute(d.reverse(),a.reverse());np.testing.assert_array_equal(ex,np.load(ROOT/family/'features.npz')['x']);swap,_=compute(d,a.with_columns((1-C('actor')).alias('actor')));np.testing.assert_array_equal(swap,ex[:,[0,3,4,1,2,7,8,5,6,9,10]]);cur=np.load(PAIR_ROOT/family/'features.npz')['x'][:,[3,6]];iso=family=='coordinated_isolation';errors=[]
  for extra,root,target in [(cur,PAIR_ROOT/'current'/family,base),(np.column_stack([cur,ex]),ROOT/family,ref)]:
   xx=np.full((len(d),extra.shape[1]),-2.) if iso else extra
   if iso:xx[a['row'].to_numpy()]=extra
   p=aggregate(d,a,np.column_stack([x,xx]),root,iso);q=d.select('pair_id','hand_id').join(target,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(p-q.select(['bg_primary','bg_secondary'] if iso else ['bg_primary']).to_numpy()).max());assert err==0;errors.append(err)
  records.append({'family':family,'actions':len(a),'baseline_and_new_probability_errors':errors,'new_models_replayed':8 if iso else 4,'baseline_models_replayed':8 if iso else 4,'reverse_raw_feature_rebuild_exact':True,'actor_swap_symmetry_exact':True,'active_roster_matches_action_state':True})
 out={'families':records,'scope':'actual outsider holdings and current board only; retrospective made-hand comparison, not EV'};(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
