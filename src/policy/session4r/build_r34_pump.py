\
\
\
\
\
import sys,csv,json,hashlib,argparse,numpy as np,polars as pl
from pathlib import Path
import os
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
ap=argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--zi',type=float,default=2.5); ap.add_argument('--za',type=float,default=3.0); ap.add_argument('--eq',type=float,default=0.55); ap.add_argument('--topk',type=int,default=7); a=ap.parse_args()
out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
ev=pl.read_parquet(S+'card_share_eval.parquet')
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(z1.alias('z1'),z2.alias('z2')).with_columns(pl.max_horizontal('z1','z2').alias('zpos'))
mem=ev.filter(((C('risk_score')>=0.5)&(C('predicted_behavior')=='coordinated_isolation')&(C('zpos')>a.zi))|((C('risk_score')>=0.05)&(C('zpos')>a.za)))
print('members',mem.height, mem['predicted_behavior'].value_counts().to_dict(as_series=False))
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','phase','started_at']).filter(C('phase')=='evaluation')
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id'])
ptab=seats.join(hands.select('hand_id','table_id'),on='hand_id').group_by('player_id').agg(C('table_id').first()); ptab=dict(zip(ptab['player_id'],ptab['table_id']))
r33={}
with open('artifacts/candidate_r33/submission.csv',newline='') as f:
    r=csv.DictReader(f)
    for row in r: r33[row['pair_id']]=row
choice={}; stats=[]
for pid,p1,p2,za_,zb_ in mem.select('pair_id','player_1','player_2','z1','z2').iter_rows():
    P,Q=(p1,p2) if za_>=zb_ else (p2,p1); t=ptab[P]
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0)
    ac=pl.read_parquet(f'artifacts/policy/actions/{t}.parquet',columns=['hand_id','player_id','street_no','action_class','action_no','phase']).filter((C('street_no')==0)&(C('phase')=='evaluation')).sort('action_no')
    fa=ac.group_by('hand_id','player_id',maintain_order=True).agg(C('action_class').first().alias('a'),C('action_no').first().alias('no'))
    j=fa.filter(C('player_id')==P).join(st.filter(C('player_id')==P).select('hand_id',C('equity').alias('eP')),on='hand_id').join(st.filter(C('player_id')==Q).select('hand_id',C('equity').alias('eQ')),on='hand_id')
    j=j.join(fa.filter(C('player_id')==Q).select('hand_id',C('no').alias('noQ')),on='hand_id',how='left').with_columns((C('noQ')>C('no')).fill_null(True).alias('q_after')).join(hands.select('hand_id','started_at'),on='hand_id')
    j=j.with_columns((C('eQ')+0.15*C('q_after').cast(pl.Float64)-0.5*(C('eP')-0.6).clip(0,None)).alias('L'))
    sel=[]
    for thr in [a.eq,0.5]:
        cand=j.filter((C('a')==3)&(C('eQ')>=thr)).sort('L',descending=True).head(a.topk).sort('started_at').head(5).sort('L',descending=True)
        sel=cand['hand_id'].to_list()
        if len(sel)>=5: break
    ncand=int(((j['a']==3)&(j['eQ']>=a.eq)).sum()); stats.append(dict(pair=pid,n_hands=j.height,n_cand=ncand,n_sel=len(sel)))
    for k in range(1,6):
        h=r33[pid][f'evidence_hand_{k}']
        if len(sel)>=5: break
        if h not in sel and h!='NO_EVIDENCE': sel.append(h)
    choice[pid]=sel[:5]
st=pl.DataFrame(stats); print(st.select(pl.exclude('pair')).describe()); print('pairs with <5 candidates at eq threshold:',int((st['n_cand']<5).sum()))
ecols=[f'evidence_hand_{k}' for k in range(1,6)]; ch=0; bch=0
with open('artifacts/candidate_r33/submission.csv',newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); hdr=next(r); w.writerow(hdr)
    for row in r:
        pid=row[0]
        if pid in choice:
            new=choice[pid]; ch+=any(x!=y for x,y in zip(row[3:8],new)); row[3:8]=new; bch+=row[2]!='other_coordination'; row[2]='other_coordination'
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()
json.dump(dict(members=mem['pair_id'].to_list(),evidence_changed=ch,behavior_changed=bch,sha256=sha,params=vars(a),stats=stats),open(out/'build_manifest.json','w'),indent=1)
print('evidence rows changed',ch,'behaviour changed',bch,'sha256',sha)
