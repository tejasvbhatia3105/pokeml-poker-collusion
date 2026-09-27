import sys
from pathlib import Path
import polars as pl
from build_candidate_r12 import geo
if __name__=='__main__':
 r12=sys.argv[1] if len(sys.argv)>1 else 'artifacts/candidate_r12/pair_eval.csv'
 v6=sys.argv[2] if len(sys.argv)>2 else 'artifacts/seq_v6/eval_all_lpo.csv'
 out=Path(sys.argv[3] if len(sys.argv)>3 else 'artifacts/candidate_r23');out.mkdir(exist_ok=True,parents=True)
 n=geo({'g':r12,'s':v6},str(out/'pair_eval.csv'),{'g':1,'s':1},pl.read_csv('data/evaluation_pairs.csv').select('pair_id'))
 print('R23 (1:1) eval >0.5',n)
