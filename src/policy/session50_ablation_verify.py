import json
import numpy as np,polars as pl
from session50_matchup import ROOT,load
from session50_verify import aggregate
def main():
 for kind in ['outcomes','current']:
  root=ROOT/kind;selected=json.load(open(root/'config.json'))['selected_columns'];ref=pl.read_parquet(root/'event_oof.parquet');records=[]
  for family in ['directed_transfer','soft_play','coordinated_isolation']:
   d,a,x,cols=load(family);ex=np.load(ROOT/family/'features.npz')['x'][:,selected];iso=family=='coordinated_isolation';xx=np.full((len(d),len(selected)),-2.) if iso else ex
   if iso:xx[a['row'].to_numpy()]=ex
   p=aggregate(d,a,np.column_stack([x,xx]),root/family,iso);q=d.select('pair_id','hand_id').join(ref,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(p-q.select(['bg_primary','bg_secondary'] if iso else ['bg_primary']).to_numpy()).max());assert err==0;records.append({'family':family,'saved_models_replayed':8 if iso else 4,'probability_error':err})
  report={'post_result_ablation':kind,'selected_columns':selected,'models':records,'raw_features':'same verified full session50 cache, explicit column slice'};(root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
