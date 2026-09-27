import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,hashlib
from pathlib import Path
import numpy as np
import polars as pl
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score,roc_auc_score
import session124_family_holdout_pairs as bench
ROOT=Path('artifacts/pair_session125_normal_only_control');C=pl.col

def main():
    ROOT.mkdir(exist_ok=True);d=pl.read_parquet(bench.ROOT/'pairs.parquet');x=np.load(bench.ROOT/'x.npy');y=d['label'].to_numpy();fv=d['fold'].to_numpy();out=[];checks=[]
    cfg=dict(method=__doc__,n_estimators=400,max_samples=256,random_seed=12501,
        training='Confirmed non-target pairs only, excluding the full query pool fold',
        calibration='Normalize raw anomaly score by median and IQR of training-normal scores',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_config=json.load(open(bench.ROOT/'data_config.json')))
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    for fold in range(4):
        tr=(fv!=fold)&(y==0);va=fv==fold;assert not set(d['table_id'].to_numpy()[tr])&set(d['table_id'].to_numpy()[va])
        m=IsolationForest(n_estimators=400,max_samples=256,random_state=12501+fold,n_jobs=2);m.fit(x[tr])
        calibration=-m.score_samples(x[tr]);mu=np.median(calibration);scale=max(1e-6,np.diff(np.quantile(calibration,[.25,.75]))[0])
        score=(-m.score_samples(x[va])-mu)/scale;path=ROOT/f'fold{fold}.joblib';joblib.dump(dict(model=m,mu=mu,scale=scale),path,compress=3)
        saved=joblib.load(path);replayed=(-saved['model'].score_samples(x[va])-saved['mu'])/saved['scale'];np.testing.assert_array_equal(score,replayed)
        out.append(d.filter(pl.Series(va)).select('pair_id','table_id','fold','label','behavior_family').with_columns(pl.Series('anomaly',score)))
        checks.append(dict(fold=fold,training_normals=int(tr.sum()),training_positives=int(y[tr].sum()),train_pair_ids=d.filter(pl.Series(tr))['pair_id'].to_list(),
            replay_error=0,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    q=pl.concat(out);q.write_parquet(ROOT/'oof.parquet');report=[]
    for family in bench.FAMILIES:
        z=q.filter((C('behavior_family')==family)|(C('label')==0));yy=z['label'].to_numpy();p=z['anomaly'].to_numpy();n=int(yy.sum())
        report.append(dict(family=family,positive_pairs=n,negative_pairs=int((yy==0).sum()),AP=average_precision_score(yy,p),AUC=roc_auc_score(yy,p),recall_at_positive_count=float(yy[np.argsort(-p,kind='stable')[:n]].mean())))
    (ROOT/'comparison.json').write_text(json.dumps(dict(results=report,limitations='Normal-only reference, not a submission model. Trusted non-target subset omits unknown background. Feature design was informed by known mechanisms. No hidden-family performance is established.'),indent=2))
    (ROOT/'verification.json').write_text(json.dumps(dict(models=checks),indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
