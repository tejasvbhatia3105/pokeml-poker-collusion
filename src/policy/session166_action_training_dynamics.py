import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session164_generic_action_em as s
from session152_generic_fallback import ap
ROOT=Path('artifacts/evidence_session166_action_training_dynamics')

def main():
    v=json.load(open(s.ROOT/'verification.json'));assert v['models']==8 and v['final_prediction_error']==0
    d,meta,x,bag,groups,aw=s.load();counts=np.bincount(bag,minlength=len(d));ranks=d['evidence_rank'].fill_null(0).to_numpy();rows=[];aps=[]
    for fold in range(4):
        p=s.math.initialize(s.old.initial(d,groups),bag)
        minimum=json.load(open(s.ROOT/f'fold{fold}_em2.json'))['minimum']
        for step in range(3):
            if step:
                m=CatBoostClassifier();m.load_model(str(s.ROOT/f'fold{fold}_em{step}.cbm'));p=m.predict_proba(x,thread_count=2)
            hp=s.math.hand_probabilities(p,bag,len(d))[0]
            for split in ['train','validation']:
                losses=[];weights=[];den=[]
                for ix in groups:
                    if (d['fold'][int(ix[0])]==fold)!=(split=='validation'):continue
                    e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])];post,ll,_=s.old.posterior(hp[ix],e)
                    if post is None:continue
                    losses.append(-ll);weights.append(1/counts[ix].sum());den.append(len(e))
                rows.append(dict(fold=fold,step=step,split=split,pairs=len(losses),pair_action_weighted_NLL=float(np.average(losses,weights=weights)),NLL_per_listed_hand=float(np.mean(np.array(losses)/den))))
            q=s.output(d,groups,bag,p,fold,minimum)
            aps.extend(dict(pair_id=pid,fold=fold,step=step,AP=ap(g)) for (pid,),g in q.group_by('pair_id'))
    ROOT.mkdir(exist_ok=True);r=pl.DataFrame(rows);a=pl.DataFrame(aps);r.write_csv(ROOT/'losses.csv');a.write_csv(ROOT/'pair_AP.csv')
    report=dict(losses=r.to_dicts(),heldout_MAP_by_step=a.group_by('step').agg(pl.col('AP').mean()).sort('step').to_dicts(),
        limitations='Post-fit diagnostic on all four completed folds. Does not change the fixed two-step endpoint, pick a submission checkpoint, or establish convergence. Unconditional observed-list likelihood matches the EM formulation; inference conditions on the training minimum count.')
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
