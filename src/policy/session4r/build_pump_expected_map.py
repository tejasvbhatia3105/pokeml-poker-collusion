\
\
\
import os,csv,json,hashlib,argparse,numpy as np,polars as pl
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--r33w',type=int,default=0); ap.add_argument('--pairs',default='77'); ap.add_argument('--eq',type=float,default=0.5); ap.add_argument('--final',default='time'); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
mem=json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']
if a.pairs=='77': mem+=list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id'])
def p_planted(eP):                                                           
    return np.interp(eP,[0.30,0.40,0.50,0.60,0.70,0.80],[0.92,0.88,0.60,0.32,0.28,0.30])
ev=pl.read_parquet(S+'card_share_eval.parquet').filter(C('pair_id').is_in(mem))
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(z1.alias('z1'),z2.alias('z2'))
files=[p for p in sorted(Path('artifacts/candidate_r33/inference').glob('*.parquet')) if not (getattr(os.stat(p),'st_flags',0) & 0x40000000)]
inf=pl.concat([pl.read_parquet(p,columns=['pair_id','hand_id','score']).filter(C('pair_id').is_in(mem)) for p in files])
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','phase','started_at']).filter(C('phase')=='evaluation')
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id']); ptab=seats.join(hands.select('hand_id','table_id'),on='hand_id').group_by('player_id').agg(C('table_id').first()); ptab=dict(zip(ptab['player_id'],ptab['table_id']))
base={}
with open(a.base,newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
choice={}; nch=[]
for pid,p1,p2,za_,zb_ in ev.select('pair_id','player_1','player_2','z1','z2').iter_rows():
    P,Q=(p1,p2) if za_>=zb_ else (p2,p1); t=ptab[P]
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0)
    ac=pl.read_parquet(f'artifacts/policy/actions/{t}.parquet',columns=['hand_id','player_id','street_no','action_class','action_no','phase']).filter((C('phase')=='evaluation')&(C('street_no')==0)&(C('player_id')==P)).sort('action_no').group_by('hand_id',maintain_order=True).agg(C('action_class').first().alias('a'))
    j=ac.filter(C('a')==3).join(st.filter(C('player_id')==Q).select('hand_id',C('equity').alias('eQ')),on='hand_id').join(st.filter(C('player_id')==P).select('hand_id',C('equity').alias('eP')),on='hand_id').filter(C('eQ')>=a.eq).join(hands.select('hand_id','started_at'),on='hand_id').sort('started_at')
    if j.is_empty(): continue
    p=p_planted(j['eP'].to_numpy())
    if a.r33w:
        sc=inf.filter(C('pair_id')==pid).select('hand_id','score'); j2=j.join(sc,on='hand_id',how='left')
        if sc.height: 
            r=j2['score'].fill_null(-1).to_numpy(); rk=(-r).argsort().argsort(); n=len(r); f=np.where(rk<n/2,1.15,0.75); p=np.clip(p*f,0.02,0.98)
                                                                        
    pl_=np.zeros(len(p)); dist=np.array([1.0])
    for i in range(len(p)):
        pl_[i]=p[i]*dist[:5].sum()
        nd=np.zeros(len(dist)+1); nd[:-1]+=dist*(1-p[i]); nd[1:]+=dist*p[i]; dist=nd
    order=np.argsort(-pl_,kind='stable')[:5]; hid=j['hand_id'].to_numpy()
    sel=[hid[i] for i in (sorted(order) if a.final=='time' else order)]
    for k in range(1,6):
        if len(sel)>=5: break
        h=base[pid][f'evidence_hand_{k}']
        if h not in sel and h!='NO_EVIDENCE': sel.append(h)
    choice[pid]=sel[:5]; nch.append(len(set(sel[:5])&{base[pid][f'evidence_hand_{k}'] for k in range(1,6)}))
ch=0
with open(a.base,newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in choice:
            new=choice[row[0]]; ch+=any(x!=y for x,y in zip(row[3:8],new)); row[3:8]=new
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print('pairs',len(choice),'rows changed',ch,'mean overlap with base',round(sum(nch)/len(nch),2),'sha',sha)
json.dump(dict(base=a.base,params=vars(a),changed=ch,sha256=sha),open(out/'build_manifest.json','w'),indent=1)
