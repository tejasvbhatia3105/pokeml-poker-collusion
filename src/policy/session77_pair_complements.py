\
\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score
ROOT=Path('artifacts/pair_session77_complements');C=pl.col
def main():
 ROOT.mkdir(exist_ok=True);q=pl.read_parquet('artifacts/evidence_session43_pair_interactions/oof.parquet');base=pl.concat([pl.read_parquet(f'artifacts/pair_session75_missing_support/{w}.parquet').select('pair_id','risk_score').with_columns(pl.lit(w).alias('window')) for w in ['full','first_2000','last_2000']]);q=q.join(base,on=['pair_id','window'],validate='1:1').with_columns(*[(C('risk_score')*C(n)).sqrt().alias('r26_'+n) for n in ['residual_only','paired_context']]);q.write_parquet(ROOT/'oof.parquet');names=['risk_score','residual_only','paired_context','r26_residual_only','r26_paired_context'];report=[]
 for (w,),d in q.group_by('window'):
  report.append({'window':w,'positive_pairs':int(d['label'].sum()),'metrics':{n:{'known_AP':average_precision_score(d['label'],d[n]),'negative_weight50_AP':average_precision_score(d['label'],d[n],sample_weight=np.where(d['label'].to_numpy()==0,50,1)),'fold_weight50_AP':[average_precision_score(z['label'],z[n],sample_weight=np.where(z['label'].to_numpy()==0,50,1)) for _,z in d.sort('fold').group_by('fold',maintain_order=True)]} for n in names}})
 (ROOT/'report.json').write_text(json.dumps({'method':__doc__,'windows':report},indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
