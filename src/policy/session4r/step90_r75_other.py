import os,polars as pl, json, csv, hashlib
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
mem=set(json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']+list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id']))
out=Path('artifacts/candidate_r75_other'); out.mkdir(exist_ok=True,parents=True); n=0
with open('artifacts/candidate_r72_evlift55/submission.csv',newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in mem and row[2]!='other_coordination': row[2]='other_coordination'; n+=1
        w.writerow(row)
print('r75 relabelled',n,'sha',hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest())
