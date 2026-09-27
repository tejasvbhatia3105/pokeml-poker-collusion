\
\
import os,csv,json,hashlib,argparse,numpy as np,polars as pl
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--mode',default='pu'); ap.add_argument('--k',type=int,default=8); ap.add_argument('--eq',type=float,default=0.5); ap.add_argument('--maxep',type=float,default=0.7); ap.add_argument('--pairs',default='65'); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
mem=json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']
if a.pairs=='77': mem+=list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id'])
pu=pl.read_parquet(S+'pump_pu_scores.parquet').filter(C('pair').is_in(mem)&(C('eQ')>=a.eq)&(C('equity')<=a.maxep))
files=[p for p in sorted(Path('artifacts/candidate_r33/inference').glob('*.parquet')) if not (getattr(os.stat(p),'st_flags',0) & 0x40000000)]
inf=pl.concat([pl.read_parquet(p,columns=['pair_id','hand_id','score']).filter(C('pair_id').is_in(mem)) for p in files]).rename({'pair_id':'pair'})
pu=pu.join(inf,on=['pair','hand_id'],how='left')
base={}
with open(a.base,newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
choice={}
for (pid,),g in pu.group_by('pair'):
    g=g.with_columns(C('pu').rank(descending=True).alias('rpu'),C('score').rank(descending=True).alias('rr33'))
    key=C('pu') if a.mode=='pu' else -(C('rpu')+pl.coalesce([C('rr33'),C('rpu')]))/2
    top=g.with_columns(key.alias('key')).sort('key',descending=True).head(a.k).sort('time_index').head(5)
    sel=top['hand_id'].to_list()
    for k in range(1,6):
        if len(sel)>=5: break
        h=base[pid][f'evidence_hand_{k}']
        if h not in sel and h!='NO_EVIDENCE': sel.append(h)
    choice[pid]=sel[:5]
ch=0
with open(a.base,newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in choice:
            new=choice[row[0]]; ch+=any(x!=y for x,y in zip(row[3:8],new)); row[3:8]=new
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print('pairs',len(choice),'rows changed',ch,'sha',sha)
json.dump(dict(base=a.base,params=vars(a),changed=ch,sha256=sha),open(out/'build_manifest.json','w'),indent=1)
