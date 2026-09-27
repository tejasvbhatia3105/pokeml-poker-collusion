import os,polars as pl, numpy as np, csv, json, hashlib
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
d=pl.read_parquet(S+'evidence_strength_pairs.parquet')
pos=d.filter((C('band')=='a>=0.9')&~C('mem'))['top5'].to_numpy(); print('confirmed positives top5 quantiles 5/10/20/50%:',np.quantile(pos,[0.05,0.1,0.2,0.5]).round(3))
low=d.filter((C('band')=='d0.05-0.2')&~C('mem'))['top5'].to_numpy(); print('band 0.05-0.2 non-members top5 quantiles 50/80/90/95/99%:',np.quantile(low,[0.5,0.8,0.9,0.95,0.99]).round(3))
band=d.filter((C('band')!='a>=0.9')&~C('mem'))
for thr in [0.55,0.6,0.62,0.65,0.7]:
    x=band.filter(C('top5')>=thr); print(f'thr {thr}: band non-member pairs above = {x.height}  (by band: {x["band"].value_counts().sort("band").to_dict(as_series=False)})  | confirmed positives below thr: {(pos<thr).mean():.3f}')
print('>=0.9 non-members with top5<0.45:',int((pos<0.45).sum()))
thr=0.62; lift=band.filter(C('top5')>=thr).select('pair_id','top5','risk33'); liftmap=dict(zip(lift['pair_id'],lift['top5']))
dem=band.filter((C('top5')<0.40)&(C('risk33')>=0.2)).select('pair_id','top5'); demmap=dict(zip(dem['pair_id'],dem['top5']))
print('lift',len(liftmap),'demote',len(demmap))
def build(out,do_demote):
    out=Path(out); out.mkdir(exist_ok=True,parents=True); nl=nd=0
    with open('artifacts/candidate_r58_77/submission.csv',newline='') as f, open(out/'submission.csv','w',newline='') as g:
        r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
        for row in r:
            pid=row[0]; rk=float(row[1])
            if pid in liftmap and rk<0.9: row[1]=repr(round(0.90+0.05*min(1,(liftmap[pid]-thr)/0.2),6)); nl+=1
            elif do_demote and pid in demmap: row[1]=repr(round(min(rk,0.04),6)); nd+=1
            w.writerow(row)
    sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print(out.name,'lifted',nl,'demoted',nd,'sha',sha)
    json.dump(dict(base='candidate_r58_77',lift_thr=thr,lifted=list(liftmap),demoted=list(demmap) if do_demote else [],sha256=sha),open(out/'build_manifest.json','w'),indent=1)
build('artifacts/candidate_r70_evlift',False)
