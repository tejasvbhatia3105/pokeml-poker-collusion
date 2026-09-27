import os,polars as pl, numpy as np, csv, json, hashlib
from pathlib import Path
from sklearn.metrics import roc_auc_score
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
d=pl.read_parquet(S+'evidence_strength_pairs.parquet'); mem=set(d.filter(C('mem'))['pair_id'])
lifted70=set(json.load(open('artifacts/candidate_r70_evlift/build_manifest.json'))['lifted'])
band=d.filter((C('band')!='a>=0.9')&~C('mem')&~C('pair_id').is_in(list(lifted70))&(C('top5')>=0.55)&(C('top5')<0.62))
l2=dict(zip(band['pair_id'],band['top5'])); print('r72 extra lifts',len(l2))
def write(base,out,lifts,lo,hi,thr_lo,thr_hi,name):
    out=Path(out); out.mkdir(exist_ok=True,parents=True); n=0
    with open(base,newline='') as f, open(out/'submission.csv','w',newline='') as g:
        r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
        for row in r:
            pid=row[0]; rk=float(row[1])
            if pid in lifts and rk<lo: row[1]=repr(round(lo+(hi-lo)*min(1,max(0,(lifts[pid]-thr_lo)/(thr_hi-thr_lo))),6)); n+=1
            w.writerow(row)
    sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(); print(name,'lifted',n,'sha',sha)
    json.dump(dict(base=base,lifted=list(lifts),range=[lo,hi],sha256=sha),open(out/'build_manifest.json','w'),indent=1)
write('artifacts/candidate_r70_evlift/submission.csv','artifacts/candidate_r72_evlift55',l2,0.86,0.90,0.55,0.62,'r72')
e=pl.read_parquet('artifacts/candidate_r30/evidence_scores.parquet')
g=e.sort(['pair_id','score'],descending=[False,True]).group_by('pair_id').agg(C('score').head(5).mean().alias('t5_30'),pl.len().alias('nh'))
r33=pl.read_csv('artifacts/candidate_r33/submission.csv').select('pair_id',C('risk_score').alias('risk33'))
x=g.join(r33,on='pair_id').with_columns(C('pair_id').is_in(list(mem)).alias('mem'))
pos=x.filter((C('risk33')>=0.9)&~C('mem'))['t5_30'].to_numpy(); b1=x.filter((C('risk33')>=0.05)&(C('risk33')<0.2)&~C('mem'))['t5_30'].to_numpy(); b0=x.filter((C('risk33')<0.05)&~C('mem'))['t5_30'].to_numpy()
print('R30-top5: positives quantiles 5/10/20/50',np.quantile(pos,[0.05,0.1,0.2,0.5]).round(3),' band0.05-0.2 q50/90/95',np.quantile(b1,[0.5,0.9,0.95]).round(3),' band0.01-0.05 q50/90/95/99',np.quantile(b0,[0.5,0.9,0.95,0.99]).round(3))
print('AUC pos vs 0.01-0.05:',round(roc_auc_score(np.r_[np.ones(len(pos)),np.zeros(len(b0))],np.r_[pos,b0]),3),' members in 0.01-0.05 band t5_30:',x.filter((C('risk33')<0.05)&C('mem'))['t5_30'].to_list())
sh=x.join(d.select('pair_id','top5'),on='pair_id'); print('corr R30-top5 vs R33-top5:',round(float(np.corrcoef(sh['t5_30'],sh['top5'])[0,1]),3))
x.write_parquet(S+'evidence_strength_r30.parquet')
