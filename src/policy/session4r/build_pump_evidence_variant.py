\
\
\
import os,csv,json,hashlib,argparse,polars as pl
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--order',default='r33'); ap.add_argument('--chrono',type=int,default=0); ap.add_argument('--eq',type=float,default=0.5); ap.add_argument('--final',default='score'); ap.add_argument('--maxep',type=float,default=1.0); ap.add_argument('--pairs',default='65'); ap.add_argument('--qafter',type=int,default=0); ap.add_argument('--pufallback',type=int,default=0); ap.add_argument('--anyraise',type=int,default=0); ap.add_argument('--fillrelax',type=float,default=0.0); ap.add_argument('--timefirst',type=int,default=0); ap.add_argument('--nprim',type=int,default=5); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
mem=json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']
if a.pairs=='77': mem+=list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id'])
if os.environ.get('PUMP_EXTRA_PAIRS'): mem=json.load(open(os.environ['PUMP_EXTRA_PAIRS']))
ev=pl.read_parquet(S+'card_share_eval.parquet').filter(C('pair_id').is_in(mem))
NMIN=int(os.environ.get('PUMP_NMIN','25'))
z1=pl.when(C('n_after').fill_null(0)>=NMIN).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=NMIN).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(z1.alias('z1'),z2.alias('z2'))
files=[p for p in sorted(Path('artifacts/candidate_r33/inference').glob('*.parquet')) if not (getattr(os.stat(p),'st_flags',0) & 0x40000000)]
inf=pl.concat([pl.read_parquet(p,columns=['pair_id','hand_id','score']).filter(C('pair_id').is_in(mem)) for p in files])
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','phase','started_at','final_pot']).filter(C('phase')=='evaluation')
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id','net_chips']); ptab=seats.join(hands.select('hand_id','table_id'),on='hand_id').group_by('player_id').agg(C('table_id').first()); ptab=dict(zip(ptab['player_id'],ptab['table_id']))
PU=pl.read_parquet(S+'pump_pu_scores.parquet') if a.pufallback else None
base={}
with open(a.base,newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
choice={}; nsc=0
for pid,p1,p2,za_,zb_ in ev.select('pair_id','player_1','player_2','z1','z2').iter_rows():
    P,Q=(p1,p2) if za_>=zb_ else (p2,p1); t=ptab[P]
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0)
    ac=pl.read_parquet(f'artifacts/policy/actions/{t}.parquet',columns=['hand_id','player_id','street_no','action_class','action_no','phase']).filter((C('street_no')==0)&(C('phase')=='evaluation')).sort('action_no')
    fa=ac.group_by('hand_id','player_id',maintain_order=True).agg((pl.when(pl.lit(bool(a.anyraise))).then((C('action_class')==3).any().cast(pl.Int32)*3).otherwise(C('action_class').first())).alias('a'),C('action_no').first().alias('no'))
    j=fa.filter(C('player_id')==P).select('hand_id','a','no').join(fa.filter(C('player_id')==Q).select('hand_id',C('no').alias('qno')),on='hand_id',how='left')
    if a.qafter: j=j.filter((C('qno')>C('no')).fill_null(True))
    j=j.select('hand_id','a').join(st.filter(C('player_id')==Q).select('hand_id',C('equity').alias('eQ')),on='hand_id').join(st.filter(C('player_id')==P).select('hand_id',C('equity').alias('eP')),on='hand_id')
    jall=j.join(hands.select('hand_id','started_at','final_pot'),on='hand_id').join(seats.filter(C('player_id')==Q).select('hand_id',C('net_chips').alias('netQ')),on='hand_id',how='left')
    j=jall.filter((C('a')==3)&(C('eQ')>=a.eq)&(C('eP')<=a.maxep))
    sc=inf.filter(C('pair_id')==pid).select('hand_id','score'); nsc+=sc.height>0
    if sc.is_empty() and PU is not None: sc=PU.filter(C('pair')==pid).select('hand_id',C('pu').alias('score'))
    j=j.join(sc,on='hand_id',how='left').with_columns(C('score').fill_null(-1.0))
    key={'r33':['score'],'pot':['final_pot','score'],'netQ':['netQ','score']}[a.order]
    if a.timefirst: j=j.sort('started_at').head(a.timefirst).sort(key,descending=True).head(5)
    elif a.chrono: j=j.sort(key,descending=True).head(a.chrono).sort('started_at').head(5)
    if a.final=='time': sel=j.sort('started_at')['hand_id'].to_list()[:a.nprim]
    else: sel=j.sort(key+['hand_id'],descending=[True]*len(key)+[False])['hand_id'].to_list()[:a.nprim]
    if a.nprim<5:                                                                                        
        fb=jall.filter(~C('hand_id').is_in(sel)).join(sc,on='hand_id',how='left').with_columns(C('score').fill_null(-1.0)).sort('score',descending=True)
        for h in fb['hand_id'].to_list():
            if len(sel)>=5: break
            if h not in sel: sel.append(h)
    if a.fillrelax and len(sel)<5:
        rel=jall.filter((C('a')==3)&(C('eQ')>=a.fillrelax)&(C('eQ')<a.eq)&(C('eP')<=a.maxep)).join(sc,on='hand_id',how='left').with_columns(C('score').fill_null(-1.0)).sort('score',descending=True).head(8).sort('started_at')
        for h in rel['hand_id'].to_list():
            if len(sel)>=5: break
            if h not in sel: sel.append(h)
        tm=dict(zip(jall['hand_id'],jall['started_at'])); sel=sorted(sel,key=lambda h: tm[h])
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
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print('pairs',len(choice),'with r33 scores',nsc,'rows changed',ch,'sha',sha)
json.dump(dict(base=a.base,params=vars(a),changed=ch,sha256=sha),open(out/'build_manifest.json','w'),indent=1)
