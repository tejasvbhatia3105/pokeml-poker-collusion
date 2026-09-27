import polars as pl, numpy as np, json, csv, hashlib, os
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
mem=set(json.load(open('artifacts/candidate_r34/build_manifest.json'))['members'])
ev=pl.read_parquet(S+'card_share_eval.parquet')
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(z1.alias('z1'),z2.alias('z2')).with_columns(pl.max_horizontal('z1','z2').alias('zpos'))
new=ev.filter((~C('pair_id').is_in(list(mem)))&(C('risk_score')>=0.05)&(C('risk_score')<0.5)&(C('zpos')>2.5)).sort('zpos',descending=True)
print('new pairs',new.height); print(new.select('pair_id','risk_score','predicted_behavior','zpos','n_after','n_after2'))
files=[p for p in sorted(Path('artifacts/candidate_r33/inference').glob('*.parquet')) if not (getattr(os.stat(p),'st_flags',0) & 0x40000000)]
inf=pl.concat([pl.read_parquet(p,columns=['pair_id','hand_id','score']).filter(C('pair_id').is_in(new['pair_id'].to_list())) for p in files])
print('R33 inference coverage:',inf['pair_id'].n_unique(),'of',new.height)
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','phase']).filter(C('phase')=='evaluation')
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id']); ptab=seats.join(hands.select('hand_id','table_id'),on='hand_id').group_by('player_id').agg(C('table_id').first()); ptab=dict(zip(ptab['player_id'],ptab['table_id']))
base={}
with open('artifacts/candidate_r40/submission.csv',newline='') as f:
    for row in csv.DictReader(f): base[row['pair_id']]=row
choice={}
for pid,p1,p2,za_,zb_ in new.select('pair_id','player_1','player_2','z1','z2').iter_rows():
    sc=inf.filter(C('pair_id')==pid)
    if sc.is_empty(): continue
    P,Q=(p1,p2) if za_>=zb_ else (p2,p1); t=ptab[P]
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0)
    ac=pl.read_parquet(f'artifacts/policy/actions/{t}.parquet',columns=['hand_id','player_id','street_no','action_class','action_no','phase']).filter((C('street_no')==0)&(C('phase')=='evaluation')).sort('action_no')
    fa=ac.group_by('hand_id','player_id',maintain_order=True).agg(C('action_class').first().alias('a'))
    j=sc.join(fa.filter(C('player_id')==P).select('hand_id','a'),on='hand_id',how='left').join(st.filter(C('player_id')==Q).select('hand_id',C('equity').alias('eQ')),on='hand_id',how='left')
    sel=j.filter((C('a')==3)&(C('eQ')>=0.5)).sort(['score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5]
    for k in range(1,6):
        if len(sel)>=5: break
        h=base[pid][f'evidence_hand_{k}']
        if h not in sel and h!='NO_EVIDENCE': sel.append(h)
    choice[pid]=sel[:5]
zmap=dict(zip(new['pair_id'],new['zpos'])); out=Path('artifacts/candidate_r41'); out.mkdir(exist_ok=True,parents=True); nb=0; ne=0
with open('artifacts/candidate_r40/submission.csv',newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        pid=row[0]
        if pid in zmap:
            row[1]=repr(round(0.85+0.01*min(zmap[pid]-2.5,4.0),6)); nb+=1
            if pid in choice: ne+=any(x!=y for x,y in zip(row[3:8],choice[pid])); row[3:8]=choice[pid]
        w.writerow(row)
sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print('boosted',nb,'evidence changed',ne,'sha',sha)
json.dump(dict(base='candidate_r40',new_pairs=new['pair_id'].to_list(),boost='0.85+0.01*min(z-2.5,4)',evidence='R33 order restricted to pump candidates',sha256=sha),open(out/'build_manifest.json','w'),indent=1)
