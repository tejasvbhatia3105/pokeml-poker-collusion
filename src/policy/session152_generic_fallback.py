\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session123_family_holdout as data

ROOT=Path('artifacts/evidence_session152_generic_fallback');C=pl.col
THRESHOLD=.5

def ap(g):
    truth=set(g.filter(C('evidence')==1)['hand_id'])
    rank=g.sort('score','hand_id',descending=[True,False])['hand_id'].to_list()[:5]
    hit=np.array([h in truth for h in rank])
    return float((hit*hit.cumsum()/np.arange(1,6)).sum()/min(5,len(truth)))

def summary(q):
    out={}
    for group in ['all','unseen','familiar']:
        z=q if group=='all' else q.filter(C('unseen')==(group=='unseen'))
        if not len(z):continue
        pool=z.group_by('table_id').agg(C('baseline','fallback').sum(),pl.len().alias('n')).sort('table_id')
        ix=np.random.default_rng(15201).integers(0,len(pool),(5000,len(pool)))
        dd=pool['fallback'].to_numpy()-pool['baseline'].to_numpy()
        boot=dd[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
        out[group]=dict(pairs=len(z),switched=int(z['switch'].sum()),baseline=z['baseline'].mean(),fallback=z['fallback'].mean(),
            delta=z['fallback'].mean()-z['baseline'].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist(),
            folds=z.group_by('fold').agg(C('baseline','fallback').mean()).sort('fold').to_dicts(),
            families=z.group_by('family').agg(C('baseline','fallback').mean(),C('switch').sum()).sort('family').to_dicts())
    return out

def main(arm):
    assert arm in ['cold','warm']
    source=Path('artifacts/evidence_session123_family_holdout' if arm=='cold' else 'artifacts/evidence_session146_warm_generic')
    verify=json.load(open(source/'verification.json'))
    assert verify['models']==32 and verify['family_exclusions_tested'] and verify['final_prediction_error']==0
    dest=ROOT/arm;dest.mkdir(parents=True,exist_ok=True)
    d,x,groups=data.load()
    support=pl.read_parquet('artifacts/evidence_session144_expert_support/expert_support.parquet')
    experts=pl.read_csv('artifacts/evidence_session142_expert_transfer/pair_comparison.csv')
    r33=pl.read_csv('artifacts/evidence_session149_persistent_donor/retrieval_comparison.csv')
    r33map=dict(r33.select('pair_id','r33').iter_rows())
    rows=[];audit=[]
    for held in ['all']+data.FAMILIES:
        generic={}
        if held=='all':
            q=pl.read_parquet(source/'all/oof.parquet')
            generic={pid:ap(g) for (pid,),g in q.group_by('pair_id')}
        else:
            for fold in range(4):
                path=source/held/f'fold{fold}_em2.cbm';record=json.load(open(path.with_suffix('.json')))
                assert record['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                assert not set(record['training_tables'])&set(d.filter(C('fold')==fold)['table_id'])
                training=d.filter(C('pair_id').is_in(record['training_pair_ids']))
                assert not training.filter(C('behavior_family')==held).height
                model=CatBoostClassifier();model.load_model(str(path))
                ix=np.flatnonzero(d['fold'].to_numpy()==fold);prob=model.predict_proba(x[ix],thread_count=2)
                q=d[ix].with_row_index('local')
                for (pid,),g in q.group_by('pair_id'):
                    g=g.sort('time_index','hand_id');pp=prob[g['local'].to_numpy(),1:]
                    generic[pid]=ap(g.with_columns(pl.Series('score',data.conditioned(pp,record['minimum']))))
                audit.append(dict(held=held,fold=fold,model_sha256=record['model_sha256'],all_outer_pairs_scored=q['pair_id'].n_unique(),pool_and_family_exclusions=True))
        assert len(generic)==372
        for r in experts.iter_rows(named=True):
            available=support.filter(C('pair_id')==r['pair_id'])
            if held!='all':available=available.filter(C('expert')!=held)
            top=available.sort('raw_top5_confidence','expert',descending=[True,False]).row(0,named=True)
            switched=top['raw_top5_confidence']<THRESHOLD
            baseline=r33map[r['pair_id']] if held=='all' else r['conditioned_'+top['expert']]
            rows.append(dict(held=held,pair_id=r['pair_id'],table_id=r['table_id'],fold=r['fold'],family=r['family'],
                unseen=held==r['family'],available_expert=top['expert'],support=top['raw_top5_confidence'],switch=switched,
                baseline=baseline,generic=generic[r['pair_id']],fallback=generic[r['pair_id']] if switched else baseline))
    q=pl.DataFrame(rows);q.write_csv(dest/'pair_comparison.csv')
    report={held:summary(q.filter(C('held')==held)) for held in ['all']+data.FAMILIES}
    transfer=q.filter(C('held')!='all');report['pooled_transfer']=summary(transfer)
    mix={}
    for held in data.FAMILIES:
        r=report[held]
        mix[held]={str(w):(1-w)*r['familiar']['delta']+w*r['unseen']['delta'] for w in [.05,.1,.2,.35,.5]}
    out=dict(method=__doc__,arm=arm,threshold=THRESHOLD,metrics=report,hypothetical_unknown_prevalence_evidence_delta=mix,
        limitations='Exploratory fixed rule on reused labels. Held-family scenarios use the confidence-selected available expert layer, not full R33. The all-family control uses actual local R33. Pooled scenario intervals cluster repeated pairs by pool. Scenario mixtures are hypothetical, not estimates of hidden prevalence or leaderboard score. No evaluation inference or submission.')
    (dest/'comparison.json').write_text(json.dumps(out,indent=2));(dest/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    import sys
    main(sys.argv[1])
