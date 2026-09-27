\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,hashlib,numpy as np,polars as pl
from scipy.stats import rankdata
from catboost import CatBoostClassifier
root=Path('artifacts/policy');dest=Path('artifacts/v5');dest.mkdir(exist_ok=True)
weight=.95;cols=json.loads((root/'residual_columns.json').read_text())
a=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='evaluation').collect().sort('pair_id');X=a.select(cols).to_numpy();pred=[]
for f in range(4):
    m=CatBoostClassifier();m.load_model(str(root/f'residual_fold{f}.cbm'));pred.append(1-m.predict_proba(X,thread_count=4)[:,0])
base=np.mean(pred,axis=0)
param=np.load(root/'novelty_parameters.npz');NX=a.select(param['columns'].tolist()).to_numpy();z=np.mean([np.log1p(abs((NX-med)/scale)) for med,scale in zip(param['med'],param['scale'])],axis=0);novel=np.sort(z,axis=1)[:,-3:].mean(1)
n=len(a);qb=rankdata(base,method='average')/(n+1);qn=rankdata(novel,method='average')/(n+1)
br=-np.log1p(-qb);nr=-np.log1p(-qn);fused=np.maximum(br,weight*nr)
q=-np.expm1(-fused)
ix=np.argsort(base,kind='stable');risk=np.interp(q,qb[ix],base[ix])
assert np.isfinite(risk).all() and (risk>=base-1e-12).all()
scores=a.select('pair_id').with_columns(pl.Series('base_risk',base),pl.Series('novelty_rarity',nr),pl.Series('base_rarity',br),pl.Series('new_risk',risk));scores.write_parquet(dest/'all_evaluation_scores.parquet',compression='zstd')
old=pl.read_csv('artifacts/v4/submission.csv');d=old.join(scores,on='pair_id',how='left',maintain_order='left')
assert d.height==old.height and d['new_risk'].null_count()==0
assert np.max(abs(d['base_risk'].to_numpy()-old['risk_score'].to_numpy()))<1e-10
promoted=(d['predicted_behavior']=='none')&(d['new_risk']>=.01)&(weight*d['novelty_rarity']>d['base_rarity'])
new=d.with_columns(pl.col('new_risk').alias('risk_score'),pl.when(promoted).then(pl.lit('other_coordination')).otherwise(pl.col('predicted_behavior')).alias('predicted_behavior')).select(old.columns)
assert new.select([c for c in old.columns if c.startswith('evidence_')]).equals(old.select([c for c in old.columns if c.startswith('evidence_')]))
new.write_csv(dest/'submission.csv')
changes=[]
for k in [100,500,1000]:
    before=set(old.sort('risk_score',descending=True).head(k)['pair_id']);after=set(new.sort('risk_score',descending=True).head(k)['pair_id']);changes.append(dict(top_k=k,new_pairs=len(after-before)))
report=dict(status='experimental_candidate_not_yet_scored_on_Kaggle',source_score=.85990,novelty_weight=weight,
    calibration_population='All evaluation-phase gameplay pairs, including pairs not requested for submission; no evaluation labels.',
    calibration_pairs=n,submission_rows=new.height,other_coordination_promotions=int(promoted.sum()),
    raised_scores=int((d['new_risk']>d['base_risk']+1e-10).sum()),ranking_changes=changes,
    evidence='Exactly the same five hands and order as v4 for every pair.',
    behavior='Retain all v4 disclosed-family predictions; promote none to other_coordination only when the novelty head raises risk past the existing 0.01 alert threshold.',
    sha256=hashlib.sha256((dest/'submission.csv').read_bytes()).hexdigest(),
    caveat='Weights selected from exploratory public-development family-withholding tests. Actual undisclosed-family and Kaggle gains remain unmeasured.')
                                                                           
manifest_path = dest/'manifest.json'
if manifest_path.exists():
    previous = json.loads(manifest_path.read_text())
    if previous.get('sha256') == report['sha256'] and 'user_reported_kaggle_score' in previous:
        for key in ['status', 'user_reported_kaggle_score', 'delta_vs_v4',
                    'inferred_pair_and_behavior_contribution', 'inferred_evidence_MAP5', 'caveat']:
            if key in previous:
                report[key] = previous[key]
manifest_path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
