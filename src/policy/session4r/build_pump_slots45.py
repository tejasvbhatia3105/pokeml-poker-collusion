\
\
\
\
\
import os,csv,json,hashlib,argparse,numpy as np,polars as pl
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--fb',default='reverse'); ap.add_argument('--nprim',type=int,default=3); ap.add_argument('--revfilter',type=int,default=0); ap.add_argument('--interleave',type=int,default=0); ap.add_argument('--revk',type=int,default=8); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
mem=json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']+list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id'])
restored=set(json.load(open('artifacts/candidate_r80_keepfam_r33ev/restored_members.json'))); mem=[m for m in mem if m not in restored]
ev=pl.read_parquet(S+'card_share_eval.parquet').filter(C('pair_id').is_in(mem))
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(z1.alias('z1'),z2.alias('z2'))
files=[p for p in sorted(Path('artifacts/candidate_r33/inference').glob('*.parquet')) if not (getattr(os.stat(p),'st_flags',0) & 0x40000000)]
inf=pl.concat([pl.read_parquet(p,columns=['pair_id','hand_id','score']).filter(C('pair_id').is_in(mem)) for p in files]); PU=pl.read_parquet(S+'pump_pu_scores.parquet')
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','phase','started_at']).filter(C('phase')=='evaluation')
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id']); ptab=seats.join(hands.select('hand_id','table_id'),on='hand_id').group_by('player_id').agg(C('table_id').first()); ptab=dict(zip(ptab['player_id'],ptab['table_id']))
base={}
with open(a.base,newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
choice={}
for pid,p1,p2,za_,zb_ in ev.select('pair_id','player_1','player_2','z1','z2').iter_rows():
    P,Q=(p1,p2) if za_>=zb_ else (p2,p1); t=ptab[P]
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0)
    ac=pl.read_parquet(f'artifacts/policy/actions/{t}.parquet',columns=['hand_id','player_id','street_no','action_class','action_no','phase']).filter((C('phase')=='evaluation')&(C('street_no')==0)).sort('action_no')
    fa=ac.group_by('hand_id','player_id',maintain_order=True).agg(C('action_class').first().alias('a'),(C('action_class')==3).any().alias('anyr'))
    eP=st.filter(C('player_id')==P).select('hand_id',C('equity').alias('eP')); eQ=st.filter(C('player_id')==Q).select('hand_id',C('equity').alias('eQ'))
    j=fa.filter(C('player_id')==P).select('hand_id','a').join(eP,on='hand_id').join(eQ,on='hand_id').join(hands.select('hand_id','started_at'),on='hand_id')
    j=j.join(fa.filter(C('player_id')==Q).select('hand_id',C('a').alias('aQ'),C('anyr').alias('qr')),on='hand_id',how='left').with_columns(C('aQ').fill_null(-1),C('qr').fill_null(False))
    sc=inf.filter(C('pair_id')==pid).select('hand_id','score')
    if sc.is_empty(): sc=PU.filter(C('pair')==pid).select('hand_id',C('pu').alias('score'))
    j=j.join(sc,on='hand_id',how='left').with_columns(C('score').fill_null(-1.0))
    cand=j.filter((C('a')==3)&(C('eQ')>=0.5)&(C('eP')<=0.7))
    prim=cand.sort('score',descending=True).head(8).sort('started_at').head(a.nprim); sel=prim['hand_id'].to_list()
    rest=j.filter(~C('hand_id').is_in(sel))
    if a.fb=='reverse':
        rv=rest.filter((C('aQ')==3)&(C('eP')>=0.5))
        if a.revfilter: rv=rv.filter(C('eQ')<=0.7).sort('score',descending=True).head(a.revk)
        fb=rv.sort('started_at')
    elif a.fb=='lowep': fb=rest.filter((C('a')==3)&(C('eQ')>=0.5)).sort('eP')
    elif a.fb=='r33q': fb=rest.filter(C('qr')&~((C('a')==3)&(C('eQ')>=0.5))).sort('score',descending=True)
    elif a.fb=='mix':
        f1=rest.filter(~((C('a')==3)&(C('eQ')>=0.5))).sort('score',descending=True).head(1); f2=rest.filter((C('aQ')==3)&(C('eP')>=0.5)&~C('hand_id').is_in(f1['hand_id'].to_list())).sort('started_at').head(1); fb=pl.concat([f1,f2])
    for h in fb['hand_id'].to_list():
        if len(sel)>=5: break
        if h not in sel: sel.append(h)
    for h in rest.filter(~((C('a')==3)&(C('eQ')>=0.5))).sort('score',descending=True)['hand_id'].to_list():
        if len(sel)>=5: break
        if h not in sel: sel.append(h)
    for k in range(1,6):
        if len(sel)>=5: break
        h=base[pid][f'evidence_hand_{k}']
        if h not in sel and h!='NO_EVIDENCE': sel.append(h)
    if a.interleave:
        tm=dict(zip(j['hand_id'],j['started_at'])); sel=sorted(sel[:5],key=lambda h: tm.get(h))
    choice[pid]=sel[:5]
ch=0
with open(a.base,newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in choice:
            new=choice[row[0]]; ch+=any(x!=y for x,y in zip(row[3:8],new)); row[3:8]=new
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print(a.fb,'pairs',len(choice),'rows changed',ch,'sha',sha)
json.dump(dict(base=a.base,params=vars(a),changed=ch,sha256=sha),open(out/'build_manifest.json','w'),indent=1)
