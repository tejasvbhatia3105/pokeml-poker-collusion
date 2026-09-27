import csv, hashlib, json, polars as pl
from pathlib import Path
C=pl.col
r77=pl.read_csv('artifacts/candidate_r77_keepfam/submission.csv'); r75=pl.read_csv('artifacts/candidate_r75_other/submission.csv')
restored=set(r77.join(r75,on='pair_id').filter(C('predicted_behavior')!=C('predicted_behavior_right'))['pair_id']); print('restored-label members',len(restored))
r33={}
with open('artifacts/candidate_r33/submission.csv',newline='') as f:
    for row in csv.DictReader(f): r33[row['pair_id']]=row
ec=[f'evidence_hand_{k}' for k in range(1,6)]; out=Path('artifacts/candidate_r80_keepfam_r33ev'); out.mkdir(exist_ok=True,parents=True); n=0
with open('artifacts/candidate_r77_keepfam/submission.csv',newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in restored:
            new=[r33[row[0]][c] for c in ec]; n+=any(x!=y for x,y in zip(row[3:8],new)); row[3:8]=new
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print('evidence rows restored to R33',n,'sha',sha)
json.dump(sorted(restored),open(out/'restored_members.json','w')); print('restored',len(restored))
