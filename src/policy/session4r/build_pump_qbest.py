\
import csv,json,hashlib,argparse,polars as pl
from pathlib import Path
import os
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--topn',type=int,default=1); ap.add_argument('--maxep',type=float,default=1.0); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
d=pl.read_parquet(S+'pump_cand_context2.parquet')
base={}
with open(a.base,newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
choice={}; nprim=[]
for (pid,),g in d.group_by('pair'):
    tm=dict(zip(g['hand_id'],g['started_at']))
    pr=g.filter((C('q_rank')<=a.topn)&(C('equity')<=a.maxep)).sort('started_at').head(5); sel=pr['hand_id'].to_list(); nprim.append(pr.height)
    for k in range(1,6):
        if len(sel)>=5: break
        h=base[pid][f'evidence_hand_{k}']
        if h not in sel and h!='NO_EVIDENCE': sel.append(h)
    import datetime; sel=sorted(sel[:5],key=lambda h: tm.get(h,datetime.datetime(2100,1,1,tzinfo=datetime.timezone.utc))); choice[pid]=sel
ch=0
with open(a.base,newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in choice:
            new=choice[row[0]]; ch+=any(x!=y for x,y in zip(row[3:8],new)); row[3:8]=new
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print('pairs',len(choice),'rows changed',ch,'primary/pair',round(sum(nprim)/len(nprim),2),'pairs>=5 primary',sum(n>=5 for n in nprim),'sha',sha)
json.dump(dict(base=a.base,params=vars(a),changed=ch,sha256=sha),open(out/'build_manifest.json','w'),indent=1)
