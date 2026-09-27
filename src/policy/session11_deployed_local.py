import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import polars as pl,numpy as np
from session8_data import hand_data
from session11_build_candidate import ROOT,NAMES,models,score
from session11_compare import compare
C=pl.col
def main():
    method=os.environ['CANDIDATE_METHOD'];m=models(method);d=hand_data();route=pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score');parts=[]
    for f in range(4):
        p=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time');q=d.filter(C('fold')==f).join(p,on=['pair_id','hand_id'],validate='1:1').join(route,on='pair_id',validate='m:1').with_columns(*[C(n).alias(f'{n}_{f}') for n in NAMES])
        for _,g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');s=score(g,f,method,m) if g['risk_score'][0]>=.05 else g['r29'].to_numpy();parts.append(g.select('pair_id','hand_id').with_columns(pl.Series(method,s)).join(g.select('pair_id','hand_id',C('r29').alias('coverage')),on=['pair_id','hand_id'],validate='1:1'))
    p=ROOT/f'{method}_deployed_oof.parquet';pl.concat(parts).write_parquet(p);os.environ['COMPARE_GATE']='.01';compare(p,[method,'coverage'],method+'_deployed')
if __name__=='__main__':main()
