import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
import build_outcome_roles as BOR
import build_relationship_evidence as BRE
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session18_order');C=pl.col
def compare(a,b):
 a=a.sort('pair_id','hand_id');b=b.sort('pair_id','hand_id');assert a.select('pair_id','hand_id').equals(b.select('pair_id','hand_id'));cols=[c for c in a.columns if c not in ['pair_id','hand_id']];x=a.select(cols).to_numpy();y=b.select(cols).to_numpy();delta=abs(x-y);return {'changed_rows':int((delta>1e-6).any(1).sum()),'max_error':float(delta.max()),'changed_columns':{c:int((delta[:,j]>1e-6).sum()) for j,c in enumerate(cols) if (delta[:,j]>1e-6).any()}}
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');q=d.select('pair_id','hand_id','table_id').join(players,on='pair_id',validate='m:1');ts=q.group_by('table_id').agg(C('pair_id').n_unique().alias('pairs')).sort('pairs','table_id',descending=[True,False])['table_id'].to_list()[:8];reports=[]
 for table in ts:
  query=q.filter(C('table_id')==table).drop('table_id').sort('pair_id','hand_id');pid=query['pair_id'][0];one=query.filter(C('pair_id')==pid);r={'table':table,'pairs':query['pair_id'].n_unique(),'hands':len(query)}
  for name,module,cache in [('outcome',BOR,'outcome_roles'),('relationship',BRE,'relationship_evidence')]:
   first=module.build(table,query,chronological=False);repeated=module.build(table,query,chronological=False);reversed_query=module.build(table,query.reverse(),chronological=False);solo=module.build(table,one,chronological=False);canonical=module.build(table,query,chronological=True);canonical_repeat=module.build(table,query.reverse(),chronological=True);reference=pl.read_parquet(f'artifacts/policy/{cache}/{table}.parquet').join(query.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');r[name]={'repeat':compare(first,repeated),'query_reversal':compare(first,reversed_query),'single_pair_batch':compare(first.filter(C('pair_id')==pid),solo),'cached_reference':compare(reference,first),'canonical_reversal':compare(canonical,canonical_repeat)}
  reports.append(r);print(table,r,flush=True)
 (ROOT/'audit.json').write_text(json.dumps(reports,indent=2))
if __name__=='__main__':main()
