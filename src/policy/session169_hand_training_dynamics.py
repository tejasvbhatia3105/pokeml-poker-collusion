import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session123_family_holdout as s
from session152_generic_fallback import ap
ROOT=Path('artifacts/evidence_session169_hand_training_dynamics')

def main():
    v=json.load(open(s.ROOT/'verification.json'));assert v['models']==32 and v['family_exclusions_tested'] and v['final_prediction_error']==0
    d,x,groups=s.load();ranks=d['evidence_rank'].fill_null(0).to_numpy();rows=[];aps=[]
    for held in ['all']+s.FAMILIES:
        for fold in range(4):
            tr,va=s.masks(d,fold,held);p=s.initial(d,groups)
            minimum=json.load(open(s.ROOT/held/f'fold{fold}_em2.json'))['minimum']
            for step in range(3):
                if step:
                    m=CatBoostClassifier();m.load_model(str(s.ROOT/held/f'fold{fold}_em{step}.cbm'));p=m.predict_proba(x,thread_count=2)
                for split,mask in [('train',tr),('validation',va)]:
                    losses=[];weights=[];den=[]
                    for ix in groups:
                        if not mask[ix[0]]:continue
                        e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])];post,ll,_=s.posterior(p[ix],e)
                        if post is None:continue
                        losses.append(-ll);weights.append(1/len(ix));den.append(len(e))
                    rows.append(dict(held=held,fold=fold,step=step,split=split,pairs=len(losses),pair_hand_weighted_NLL=float(np.average(losses,weights=weights)),NLL_per_listed_hand=float(np.mean(np.array(losses)/den))))
                for ix in groups:
                    if not va[ix[0]]:continue
                    g=d[ix].with_columns(pl.Series('score',s.conditioned(p[ix,1:],minimum)))
                    aps.append(dict(held=held,pair_id=g['pair_id'][0],fold=fold,step=step,AP=ap(g)))
    ROOT.mkdir(exist_ok=True);r=pl.DataFrame(rows);a=pl.DataFrame(aps);r.write_csv(ROOT/'losses.csv');a.write_csv(ROOT/'pair_AP.csv')
    out=dict(losses=r.to_dicts(),heldout_MAP_by_arm_step=a.group_by('held','step').agg(pl.col('AP').mean()).sort('held','step').to_dicts(),
        limitations='Post-fit diagnostic on all completed folds and original cold123 configurations. No checkpoint selected, model retrained, convergence asserted, or leaderboard improvement inferred.')
    (ROOT/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
