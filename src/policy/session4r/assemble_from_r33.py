\
import sys,csv,hashlib,argparse,polars as pl
C=pl.col; N=['directed_transfer','soft_play','coordinated_isolation']
ap=argparse.ArgumentParser(); ap.add_argument('pair_eval'); ap.add_argument('out'); ap.add_argument('--behavior-min',type=float,default=0.01); ap.add_argument('--base',default='artifacts/candidate_r33/submission.csv'); a=ap.parse_args()
p=pl.read_csv(a.pair_eval).with_columns((1-C('none')).alias('risk')); fam=[N[k] for k in p.select(N).to_numpy().argmax(1)]
risk=dict(zip(p['pair_id'],p['risk'])); famm=dict(zip(p['pair_id'],fam))
with open(a.base,newline='') as f, open(a.out,'w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); hdr=next(r); w.writerow(hdr); n=0; ch=0; bch=0
    for row in r:
        pid=row[0]; rk=risk[pid]; beh=row[2] if row[2]=='other_coordination' else (famm[pid] if rk>=a.behavior_min else 'none')
        ch+=float(row[1])!=rk; bch+=row[2]!=beh; row[1]=repr(float(rk)); row[2]=beh; w.writerow(row); n+=1
print('rows',n,'risk changed',ch,'behavior changed',bch,'sha256',hashlib.sha256(open(a.out,'rb').read()).hexdigest())
