import os,json,sys
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import polars as pl,numpy as np
from session8_data import hand_data
from session6_priority import inclusion
C=pl.col
def main():
 root=Path(sys.argv[1]);d=hand_data();old=pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet').select('pair_id','hand_id','primary','secondary');q=d.join(old,on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet(root/'event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1');rows=[]
 for (pid,),g in q.group_by('pair_id'):
  g=g.sort('time','hand_id');truth=set(g.filter(C('evidence')==1)['hand_id']);row={'pair_id':pid,'fold':g['fold'][0],'family':g['behavior_family'][0]}
  for name,a,b in [('old',g['primary'].to_numpy(),g['secondary'].to_numpy()),('new',g['bg_primary'].to_numpy(),g['bg_secondary'].to_numpy())]:
   s=inclusion(a,b);rank=np.lexsort((g['hand_id'].to_numpy(),-s))[:5];y=np.array([g['hand_id'][int(i)] in truth for i in rank]);row[name]=float((y*y.cumsum()/np.arange(1,len(y)+1)).sum()/min(5,len(truth)))
  rows.append(row)
 r=pl.DataFrame(rows);r.write_csv(root/'standalone_comparison.csv');report={'overall':r.select(C('old','new').mean()).to_dicts()[0],'families':r.group_by('family').agg(C('old','new').mean()).to_dicts(),'folds':r.group_by('fold').agg(C('old','new').mean()).sort('fold').to_dicts()};(root/'standalone_comparison.json').write_text(json.dumps(report,indent=2));print(root,report)
if __name__=='__main__':main()
