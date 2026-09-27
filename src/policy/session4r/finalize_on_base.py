import sys,csv,json,hashlib
from pathlib import Path
src,out,basep=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]; out.mkdir(exist_ok=True,parents=True)
restored=set(json.load(open('artifacts/candidate_r80_keepfam_r33ev/restored_members.json'))); base={}
with open(basep,newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
ec=[f'evidence_hand_{k}' for k in range(1,6)]; n=0
with open(src/'submission.csv',newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in restored: row[3:8]=[base[row[0]][c] for c in ec]
        elif any(row[3+i]!=base[row[0]][ec[i]] for i in range(5)): n+=1
        w.writerow(row)
print(out.name,'pump rows changed vs base:',n,'sha',hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest())
