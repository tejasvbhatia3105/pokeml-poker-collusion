import json
from pathlib import Path
import numpy as np,polars as pl
import session88_crop_audit as s
C=pl.col
def main():
 d=s.data();truth=pl.read_csv('data/development_evidence.csv');cov=pl.read_parquet(s.ROOT/'coverage.parquet');count=0;checks=0
 for (table,),g in d.group_by('table_id'):
  raw=pl.read_parquet(list((Path('artifacts/compact/hands')/f'table_id={table}').glob('*.parquet')),columns=['hand_id','started_at','phase']).sort('started_at','hand_id').with_row_index('raw_index');q=g.select('pair_id','hand_id','time').join(raw,on='hand_id',validate='m:1');np.testing.assert_array_equal(np.rint(q['time'].to_numpy()*5000).astype(int),q['raw_index'].to_numpy());assert (q['phase']=='development').all();count+=len(q)
  for (pid,),z in q.group_by('pair_id'):
   published=set(truth.filter(C('pair_id')==pid)['hand_id']);assert published<=set(z['hand_id'])
   for w,(lo,hi) in s.WINDOWS.items():
    kept=set(z.filter((C('raw_index')>=round(lo*5000))&(C('raw_index')<round(hi*5000)))['hand_id']);r=cov.filter((C('pair_id')==pid)&(C('window')==w)).row(0,named=True);assert r['truth_retained']==len(kept&published);assert r['complete_list_retained']==published.issubset(kept);assert r['hands']==len(kept);checks+=1
 report={'hand_times_verified_against_raw_timestamps':count,'raw_time_index_error':0,'published_truth_crop_coverage_checks':checks,'coverage_error':0,'full_R33_replay_error':json.load(open(s.ROOT/'report.json'))['full_R33_score_replay_error'],'limits':'This verifies list-stage cropping, not recomputed styles or private phase truth.'};(s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
