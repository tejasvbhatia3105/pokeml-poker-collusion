\
\
import sys,os,csv,json,hashlib,argparse,polars as pl
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--eq',type=float,default=0.5); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
m=json.load(open('artifacts/candidate_r34/build_manifest.json')); mem=m['members']
ev=pl.read_parquet(S+'card_share_eval.parquet').filter(C('pair_id').is_in(mem))
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(z1.alias('z1'),z2.alias('z2'))
files=[p for p in sorted(Path('artifacts/candidate_r33/inference').glob('*.parquet')) if not (getattr(os.stat(p),'st_flags',0) & 0x40000000)]
inf=pl.concat([pl.read_parquet(p,columns=['pair_id','hand_id','score']).filter(C('pair_id').is_in(mem)) for p in files])
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','phase']).filter(C('phase')=='evaluation')
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id']); ptab=seats.join(hands.select('hand_id','table_id'),on='hand_id').group_by('player_id').agg(C('table_id').first()); ptab=dict(zip(ptab['player_id'],ptab['table_id']))
r33={}
with open(a.base,newline='') as f:
    for row in csv.DictReader(f): r33[row['pair_id']]=row
choice={}; stats=[]
for pid,p1,p2,za_,zb_ in ev.select('pair_id','player_1','player_2','z1','z2').iter_rows():
    sc=inf.filter(C('pair_id')==pid)
    if sc.is_empty(): stats.append(dict(pair=pid,status='no_r33_scores')); continue
    P,Q=(p1,p2) if za_>=zb_ else (p2,p1); t=ptab[P]
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0)
    ac=pl.read_parquet(f'artifacts/policy/actions/{t}.parquet',columns=['hand_id','player_id','street_no','action_class','action_no','phase']).filter((C('street_no')==0)&(C('phase')=='evaluation')).sort('action_no')
    fa=ac.group_by('hand_id','player_id',maintain_order=True).agg(C('action_class').first().alias('a'))
    j=sc.join(fa.filter(C('player_id')==P).select('hand_id','a'),on='hand_id',how='left').join(st.filter(C('player_id')==Q).select('hand_id',C('equity').alias('eQ')),on='hand_id',how='left')
    cand=j.filter((C('a')==3)&(C('eQ')>=a.eq)).sort(['score','hand_id'],descending=[True,False])
    sel=cand['hand_id'].to_list()[:5]
    for k in range(1,6):
        if len(sel)>=5: break
        h=r33[pid][f'evidence_hand_{k}']
        if h not in sel and h!='NO_EVIDENCE': sel.append(h)
    choice[pid]=sel[:5]; stats.append(dict(pair=pid,status='ok',n_cand=cand.height,overlap_r33=len(set(sel)&{r33[pid][f'evidence_hand_{k}'] for k in range(1,6)})))
ecols=[f'evidence_hand_{k}' for k in range(1,6)]; ch=0
with open(a.base,newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in choice:
            new=choice[row[0]]; ch+=any(x!=y for x,y in zip(row[3:8],new)); row[3:8]=new
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()
ok=[s for s in stats if s['status']=='ok']; print('members',len(mem),'scored',len(ok),'evidence rows changed',ch,'mean overlap with base',sum(s['overlap_r33'] for s in ok)/max(1,len(ok)),'mean cand',sum(s['n_cand'] for s in ok)/max(1,len(ok)),'sha',sha)
json.dump(dict(base=a.base,eq=a.eq,changed=ch,sha256=sha,stats=stats),open(out/'build_manifest.json','w'),indent=1)
