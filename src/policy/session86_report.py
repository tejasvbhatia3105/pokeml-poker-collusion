import json
import numpy as np,polars as pl
from session86_action_history import ROOT
def main():
 z=np.load(ROOT/'sample.npz');meta=pl.read_parquet(ROOT/'sample_keys.parquet');fv=z['fold'];y=z['y'];parts=[]
 for f in range(4):
  va=fv==f;q=meta.filter(pl.Series(va));loss={}
  for kind in ['reference','summary','history']:
   p=np.load(ROOT/f'{kind}_fold{f}.npy');loss[kind]=-np.log(p[np.arange(len(p)),y[va]].clip(1e-7)).astype(float)
  parts.append(q.with_columns(pl.lit(f).alias('fold'),*[pl.Series(k,v) for k,v in loss.items()]))
 q=pl.concat(parts);cols=['reference','summary','history'];pool=q.group_by('table_id').agg(pl.col(cols).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(8612).integers(0,len(pool),(5000,len(pool)));diff=pool['history'].to_numpy()-pool['summary'].to_numpy();b=diff[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report={'metric':'heldout gameplay action logloss, lower is better; not evidence MAP or competition score','actions':len(q),'pools':len(pool),'pooled_logloss':{k:q[k].mean() for k in cols},'fold_logloss':q.group_by('fold').agg(pl.col(cols).mean()).sort('fold').to_dicts(),'history_minus_summary':float(q['history'].mean()-q['summary'].mean()),'pool_CI95_history_minus_summary':np.quantile(b,[.025,.975]).tolist(),'schedule':'fixed12epochs; no checkpoint selection','evidence_labels_used':False};(ROOT/'comparison.json').write_text(json.dumps(report,indent=2));pool.write_parquet(ROOT/'pool_losses.parquet');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
