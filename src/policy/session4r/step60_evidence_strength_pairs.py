import polars as pl, numpy as np, os, json
from pathlib import Path
from sklearn.metrics import roc_auc_score
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
files=[p for p in sorted(Path('artifacts/candidate_r33/inference').glob('*.parquet')) if not (getattr(os.stat(p),'st_flags',0) & 0x40000000)]
inf=pl.concat([pl.read_parquet(p,columns=['pair_id','hand_id','time','behavior_family','score']) for p in files])
g=inf.sort(['pair_id','score'],descending=[False,True]).group_by('pair_id').agg(pl.len().alias('n_hands'),C('score').head(5).mean().alias('top5'),C('score').head(1).mean().alias('top1'),C('score').head(10).mean().alias('top10'),(C('score')>0.5).sum().alias('n_gt05'),(C('score')>0.3).sum().alias('n_gt03'),C('score').mean().alias('mean_score'),C('behavior_family').first().alias('fam'))
sub=pl.read_csv('artifacts/candidate_r58_77/submission.csv').select('pair_id','risk_score')
r33=pl.read_csv('artifacts/candidate_r33/submission.csv').select('pair_id',C('risk_score').alias('risk33'))
mem=set(json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']+list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id']))
d=g.join(sub,on='pair_id').join(r33,on='pair_id').with_columns(C('pair_id').is_in(list(mem)).alias('mem'))
d=d.with_columns(pl.when(C('risk33')>=0.9).then(pl.lit('a>=0.9')).when(C('risk33')>=0.5).then(pl.lit('b0.5-0.9')).when(C('risk33')>=0.2).then(pl.lit('c0.2-0.5')).otherwise(pl.lit('d0.05-0.2')).alias('band'))
pl.Config.set_tbl_width_chars(220)
print(d.group_by('band','mem').agg(pl.len(),C('top5').mean().alias('top5_mean'),C('top5').median().alias('top5_med'),C('top1').mean().alias('top1_mean'),C('n_gt05').mean().alias('ngt05'),C('n_gt03').mean().alias('ngt03'),C('mean_score').mean().alias('meansc'),C('n_hands').mean().alias('nh')).sort('band','mem'))
pos=d.filter((C('band')=='a>=0.9')&~C('mem')); low=d.filter((C('band')=='d0.05-0.2')&~C('mem'))
for f in ['top5','top1','top10','n_gt05','n_gt03','mean_score']:
    y=np.r_[np.ones(pos.height),np.zeros(low.height)]; x=np.r_[pos[f].to_numpy(),low[f].to_numpy()]; print(f'{f:10s} AUC positives(>=0.9) vs band 0.05-0.2: {roc_auc_score(y,x):.3f}')
for b in ['b0.5-0.9','c0.2-0.5','d0.05-0.2']:
    m=d.filter((C('band')==b)&C('mem')); n=d.filter((C('band')==b)&~C('mem'))
    if m.height: print(b,'members top5 mean',round(float(m['top5'].mean()),3),'non-members',round(float(n['top5'].mean()),3),' members n_gt05',round(float(m['n_gt05'].mean()),2),'non',round(float(n['n_gt05'].mean()),2))
d.write_parquet(S+'evidence_strength_pairs.parquet')
