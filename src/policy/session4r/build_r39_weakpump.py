\
\
\
\
import csv,json,hashlib,argparse,polars as pl
from pathlib import Path
import os
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ep',type=float,default=0.55); ap.add_argument('--eq',type=float,default=0.55); ap.add_argument('--order',default='eP'); ap.add_argument('--base',default='artifacts/candidate_r33/submission.csv'); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
d=pl.read_parquet(S+'pump_hand_detail.parquet').join(pl.read_parquet('data/hands.parquet',columns=['hand_id','started_at']),on='hand_id')
base={}
with open(a.base,newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
choice={}; stats=[]
for (pid,),g in d.group_by('pair'):
    sel=[]
    for ep in [a.ep,0.6,0.7]:
        c=g.filter((C('Pfirst')=='r')&(C('eQ')>=a.eq)&(C('eP')<ep)).sort('started_at').head(5)
        c=c.sort('eP') if a.order=='eP' else c.sort('started_at')
        sel=c['hand_id'].to_list()
        if len(sel)>=5: break
    ncand=int(((g['Pfirst']=='r')&(g['eQ']>=a.eq)&(g['eP']<a.ep)).sum())
    for k in range(1,6):
        if len(sel)>=5: break
        h=base[pid][f'evidence_hand_{k}']
        if h not in sel and h!='NO_EVIDENCE': sel.append(h)
    choice[pid]=sel[:5]; stats.append(dict(pair=pid,n_cand=ncand,overlap_base=len(set(sel[:5])&{base[pid][f'evidence_hand_{k}'] for k in range(1,6)})))
ch=0
with open(a.base,newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in choice:
            new=choice[row[0]]; ch+=any(x!=y for x,y in zip(row[3:8],new)); row[3:8]=new
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()
st=pl.DataFrame(stats); print('pairs',len(choice),'rows changed',ch,'n_cand mean',float(st['n_cand'].mean()),'pairs with <5 cand',int((st['n_cand']<5).sum()),'>5',int((st['n_cand']>5).sum()),'mean overlap with base',float(st['overlap_base'].mean()),'sha',sha)
json.dump(dict(params=vars(a),changed=ch,sha256=sha,stats=stats),open(out/'build_manifest.json','w'),indent=1)
