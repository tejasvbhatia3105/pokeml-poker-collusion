import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session180_cohort_conditioned_action_em as model
from session152_generic_fallback import ap
s=model.s;ROOT=Path('artifacts/evidence_session187_cohort_failure_audit');C=pl.col

def main():
    assert json.load(open(model.ROOT/'verification.json'))['models']==24
    d,meta,x,bag,groups,aw=s.load();saved=pl.read_parquet(model.ROOT/'oof.parquet');ranks=d['evidence_rank'].fill_null(0).to_numpy();initial=s.old.initial(d,groups);rows=[];checks=[]
    counts=np.bincount(bag,minlength=len(d))
    for fold in range(4):
        path=model.ROOT/f'fold{fold}_em6.cbm';record=json.load(open(path.with_suffix('.json')));assert record['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        m=CatBoostClassifier();m.load_model(str(path));assert np.array_equal(m.classes_,[0,1,2,3]);raw=m.predict_proba(x,thread_count=2);p=model.conditional(raw);np.testing.assert_allclose(p.sum(1),1,atol=1e-15)
        hp=s.math.hand_probabilities(p,bag,len(d))[0];cohort=np.bincount(bag,weights=raw[:,1:].sum(1),minlength=len(d));scores=np.zeros(len(d))
        for ix in groups:
            e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])];compatible=s.old.posterior(initial[ix],e)[0] is not None
            held=d['fold'][int(ix[0])]==fold;split='heldout' if held else ('train' if compatible else 'omitted_train_pool')
            scores[ix]=s.old.conditioned(hp[ix,1:],record['minimum']);g=d[ix].with_columns(pl.Series('score',scores[ix]))
            rows.append(dict(fit_fold=fold,pair_id=g['pair_id'][0],table_id=g['table_id'][0],family=g['behavior_family'][0],split=split,AP=ap(g),
                mean_positive_cohort_probability=float(cohort[ix].sum()/counts[ix].sum()),expected_hand_events=float(hp[ix,1:].sum()),hands=len(ix)))
        check=d.with_columns(pl.Series('score',scores)).filter(C('fold')==fold).select('pair_id','hand_id','score').join(saved.select('pair_id','hand_id',C('score').alias('saved')),on=['pair_id','hand_id'],validate='1:1')
        error=float((check['score']-check['saved']).abs().max());assert error==0;checks.append(dict(fold=fold,heldout_replay_error=error,classes=[0,1,2,3],conditional_sums_one=True))
    q=pl.DataFrame(rows);ROOT.mkdir(exist_ok=True);q.write_csv(ROOT/'pair_fit.csv');summary=q.group_by('split').agg(C('AP','mean_positive_cohort_probability','expected_hand_events').mean(),pl.len().alias('pair_predictions')).sort('split')
    out=dict(summary=summary.to_dicts(),checks=checks,limitations='Training fit is not validation. The positive-cohort probability is an auxiliary action-level estimate, not a calibrated relationship risk. No inference branch is chosen with query labels; every query uses the same normalized positive-cohort distribution. No fit, repair, candidate or causal attribution.')
    (ROOT/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
