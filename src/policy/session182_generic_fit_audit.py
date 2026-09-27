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
import session123_family_holdout as hand
import session164_generic_action_em as action
from session152_generic_fallback import ap
ROOT=Path('artifacts/evidence_session182_generic_fit_audit');C=pl.col
VARIANTS={
    'cold_hand':('artifacts/evidence_session123_family_holdout','hand',2),
    'direct_membership':('artifacts/evidence_session174_direct_list_membership','direct',0),
    'direct_outcomes':('artifacts/evidence_session178_retrospective_membership','outcomes',0),
    'action_em2':('artifacts/evidence_session164_generic_action_em','action',2),
    'action_em6':('artifacts/evidence_session167_action_em_extended','action',6),
    'negative_action':('artifacts/evidence_session172_negative_action_em','action',6),
}

def main():
    ROOT.mkdir(exist_ok=True);d,hx,groups=hand.load();ad,meta,ax,bag,agroups,aw=action.load();assert d.equals(ad)
    extra=np.load('artifacts/evidence_session177_retrospective_features/extra.npy',mmap_mode='r');rows=[];audit=[]
    initial=hand.initial(d,groups);ranks=d['evidence_rank'].fill_null(0).to_numpy();compatible=np.zeros(len(d),bool)
    for ix in groups:
        e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])]
        compatible[ix]=hand.posterior(initial[ix],e)[0] is not None
    for name,(root,kind,step) in VARIANTS.items():
        root=Path(root);verification=json.load(open(root/'verification.json'));assert verification['final_prediction_error']==0
        folder=root if kind=='action' else root/'all';saved=pl.read_parquet(folder/'oof.parquet')
        for fold in range(4):
            stem=f'fold{fold}_em{step}' if step else f'fold{fold}';path=folder/(stem+'.cbm');record=json.load(open(path.with_suffix('.json')))
            assert record['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            m=CatBoostClassifier();m.load_model(str(path));x=ax if kind=='action' else (np.column_stack([hx,extra]) if kind=='outcomes' else hx)
            p=m.predict_proba(x,thread_count=2)
            if kind=='action':p=action.math.hand_probabilities(p,bag,len(d))[0]
            scores=np.empty(len(d));tr=d['fold'].to_numpy()!=fold
            for ix in groups:
                scores[ix]=p[ix,1] if kind in ['direct','outcomes'] else hand.conditioned(p[ix,1:],record['minimum'])
                split='heldout' if not tr[ix[0]] else ('train' if kind in ['direct','outcomes'] or compatible[ix[0]] else 'omitted_train_pool')
                g=d[ix].with_columns(pl.Series('score',scores[ix]));rows.append(dict(model=name,fit_fold=fold,split=split,pair_id=g['pair_id'][0],table_id=g['table_id'][0],family=g['behavior_family'][0],AP=ap(g)))
            omitted=sum(bool(tr[ix[0]] and not compatible[ix[0]]) for ix in groups) if kind not in ['direct','outcomes'] else 0
            assert omitted==record.get('excluded_incompatible_pairs',record.get('incompatible_training_pairs',0))
            check=d.with_columns(pl.Series('score',scores)).filter(C('fold')==fold).select('pair_id','hand_id','score').join(saved.select('pair_id','hand_id',C('score').alias('saved')),on=['pair_id','hand_id'],validate='1:1')
            error=float((check['score']-check['saved']).abs().max());assert error==0
            audit.append(dict(model=name,fold=fold,model_sha256=record['model_sha256'],heldout_score_replay_error=error,omitted_training_pairs=omitted))
        print('FIT_AUDIT',name,flush=True)
    q=pl.DataFrame(rows);q.write_csv(ROOT/'pair_fit.csv');summary=q.group_by('model','split').agg(C('AP').mean(),pl.len().alias('pair_predictions')).sort('model','split')
    result=dict(summary=summary.to_dicts(),by_fold=q.group_by('model','fit_fold','split').agg(C('AP').mean()).sort('model','fit_fold','split').to_dicts(),audit=audit,limitations=__doc__)
    (ROOT/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
