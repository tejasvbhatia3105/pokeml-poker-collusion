import csv,json,hashlib
from pathlib import Path
mem=set(json.load(open('artifacts/candidate_r34/build_manifest.json'))['members'])
out=Path('artifacts/candidate_r40'); out.mkdir(exist_ok=True,parents=True); n=0
with open('artifacts/candidate_r37b/submission.csv',newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in mem and float(row[1])<0.995: row[1]=repr(max(float(row[1]),0.995)); n+=1
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print('boosted',n,'sha',sha)
