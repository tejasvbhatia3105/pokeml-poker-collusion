\
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
from scipy.stats import rankdata
from session75_missing_support import MODELS,WEIGHTS
from session73_exclusion_audit import metrics
ROOT=Path('artifacts/pair_session76_rank_pool');C=pl.col
def rank_pool(x,baseline):
 ranks=np.column_stack([rankdata(-x[:,k],method='average') for k in range(x.shape[1])]);latent=-np.log((ranks-.5)/len(x))@WEIGHTS;order=rankdata(latent,method='average')-1;mapped=np.interp(order,np.arange(len(x)),np.sort(baseline));return mapped,latent
def main():
 ROOT.mkdir(exist_ok=True);report=[]
 for w in ['full','first_2000','last_2000']:
  q=pl.read_parquet(f'artifacts/pair_session75_missing_support/{w}.parquet');x=q.select([v+'_lpo' for v in MODELS]).to_numpy();base=q['risk_score'].to_numpy();p,latent=rank_pool(x,base);pp,ll=rank_pool(x[::-1],base[::-1]);np.testing.assert_array_equal(p,pp[::-1]);np.testing.assert_array_equal(latent,ll[::-1]);q=q.with_columns(pl.Series('tail_rank_pool',p));q.select('pair_id','tail_rank_pool').write_parquet(ROOT/f'{w}.parquet');report.append({'window':w,'metrics':{n:metrics(q,n) for n in ['risk_score','tail_rank_pool']},'pairs_crossing_005_up':q.filter((C('risk_score')<.05)&(C('tail_rank_pool')>=.05)).height,'pairs_crossing_005_down':q.filter((C('risk_score')>=.05)&(C('tail_rank_pool')<.05)).height,'row_permutation_exact':True})
 (ROOT/'report.json').write_text(json.dumps({'method':__doc__,'windows':report},indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
