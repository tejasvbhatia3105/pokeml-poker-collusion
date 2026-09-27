\
\
\
import os,csv,json,hashlib,argparse,numpy as np,polars as pl
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--primary',default='qraise'); ap.add_argument('--eq',type=float,default=0.5); ap.add_argument('--k',type=int,default=8); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
mem=json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']+list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id'])
ev=pl.read_parquet(S+'card_share_eval.parquet').filter(C('pair_id').is_in(mem))
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(z1.alias('z1'),z2.alias('z2'))
files=[p for p in sorted(Path('artifacts/candidate_r33/inference').glob('*.parquet')) if not (getattr(os.stat(p),'st_flags',0) & 0x40000000)]
inf=pl.concat([pl.read_parquet(p,columns=['pair_id','hand_id','score']).filter(C('pair_id').is_in(mem)) for p in files])
PU=pl.read_parquet(S+'pump_pu_scores.parquet')
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','phase','started_at']).filter(C('phase')=='evaluation')
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id','folded']); ptab=seats.join(hands.select('hand_id','table_id'),on='hand_id').group_by('player_id').agg(C('table_id').first()); ptab=dict(zip(ptab['player_id'],ptab['table_id']))
base={}
with open(a.base,newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
choice={}; nprim=[]
for pid,p1,p2,za_,zb_ in ev.select('pair_id','player_1','player_2','z1','z2').iter_rows():
    P,Q=(p1,p2) if za_>=zb_ else (p2,p1); t=ptab[P]
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0)
    ac=pl.read_parquet(f'artifacts/policy/actions/{t}.parquet',columns=['hand_id','player_id','street_no','action_class','action_no','phase']).filter((C('phase')=='evaluation')&(C('street_no')==0)).sort('action_no')
    pf=ac.filter(C('player_id')==P).group_by('hand_id',maintain_order=True).agg(C('action_class').first().alias('a'))
    qf=ac.filter(C('player_id')==Q).group_by('hand_id').agg((C('action_class')==3).any().alias('q_raised'),(C('action_class')==0).any().alias('q_folded_pf'))
    j=pf.filter(C('a')==3).join(st.filter(C('player_id')==Q).select('hand_id',C('equity').alias('eQ')),on='hand_id').filter(C('eQ')>=a.eq).join(qf,on='hand_id',how='left').with_columns(C('q_raised').fill_null(False),C('q_folded_pf').fill_null(True))
    j=j.join(seats.filter(C('player_id')==P).select('hand_id',C('folded').alias('p_folded')),on='hand_id').join(hands.select('hand_id','started_at'),on='hand_id')
    sc=inf.filter(C('pair_id')==pid).select('hand_id','score')
    if sc.is_empty(): sc=PU.filter(C('pair')==pid).select('hand_id',C('pu').alias('score'))
    j=j.join(sc,on='hand_id',how='left').with_columns(C('score').fill_null(-1.0))
    prim={'qraise':C('q_raised'),'qplay':~C('q_folded_pf'),'both':C('q_raised')&~C('p_folded')}[a.primary]
    pr=j.filter(prim).sort('started_at').head(5); sel=pr['hand_id'].to_list(); nprim.append(pr.height)
    if len(sel)<5:
        fb=j.filter(~prim).sort('score',descending=True).head(a.k).sort('started_at')
        for h in fb['hand_id'].to_list():
            if len(sel)>=5: break
            sel.append(h)
                       
    tm=dict(zip(j['hand_id'],j['started_at'])); sel=sorted(sel,key=lambda h: tm.get(h))
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
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print('pairs',len(choice),'rows changed',ch,'primary count/pair mean',round(float(np.mean(nprim)),2),'pairs with >=5 primary',int(np.sum(np.array(nprim)>=5)),'sha',sha)
json.dump(dict(base=a.base,params=vars(a),changed=ch,sha256=sha),open(out/'build_manifest.json','w'),indent=1)
