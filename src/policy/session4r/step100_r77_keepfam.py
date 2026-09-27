import os,polars as pl, numpy as np, json, csv, hashlib
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
mem=json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']+list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id'])
p=pl.read_csv('artifacts/candidate_r26/pair_eval.csv').with_columns((1-C('none')).alias('risk')).filter(C('pair_id').is_in(mem))
N=['directed_transfer','soft_play','coordinated_isolation']; sh=p.select(N).to_numpy()/p['risk'].to_numpy()[:,None]; mx=sh.max(1); fam=np.array(N)[sh.argmax(1)]
ev=pl.read_parquet(S+'card_share_eval.parquet').filter(C('pair_id').is_in(mem))
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
zz=dict(zip(ev['pair_id'],ev.with_columns(pl.max_horizontal(z1,z2).alias('zpos'))['zpos']))
d=pl.DataFrame({'pair_id':p['pair_id'],'fam':fam,'share':mx}).with_columns(pl.Series('zpos',[zz.get(x,0) for x in p['pair_id']]))
print('members family-share quantiles',np.quantile(mx,[0.1,0.25,0.5,0.75,0.9]).round(3))
for thr in [0.95,0.98,0.99,0.995]: print(f'share>={thr}: {int((mx>=thr).sum())} members  (fams {d.filter(C("share")>=thr)["fam"].value_counts().to_dict(as_series=False)})')
d=d.with_columns((C('share')-0.05*C('zpos')).alias('keep_score')); print(d.sort('keep_score',descending=True).head(12))
keep=set(d.filter((C('share')>=0.98)&(C('zpos')<3.5))['pair_id']); kf=dict(zip(d['pair_id'],d['fam'])); print('keep family label for',len(keep))
out=Path('artifacts/candidate_r77_keepfam'); out.mkdir(exist_ok=True,parents=True); n=0
with open('artifacts/candidate_r75_other/submission.csv',newline='') as f, open(out/'submission.csv','w',newline='') as g:
    r=csv.reader(f); w=csv.writer(g,lineterminator='\n'); w.writerow(next(r))
    for row in r:
        if row[0] in keep: row[2]=kf[row[0]]; n+=1
        w.writerow(row)
print('r77 relabelled back',n,'sha',hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest())
