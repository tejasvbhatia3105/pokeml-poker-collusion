\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
import session164_generic_action_em as s
ROOT=Path('artifacts/evidence_session186_action_mstep_audit');SOURCE=Path('artifacts/evidence_session167_action_em_extended')

def main():
    assert json.load(open(SOURCE/'verification.json'))['models']==24
    d,meta,x,bag,groups,aw=s.load();rows_out=[];ranks=d['evidence_rank'].fill_null(0).to_numpy();counts=np.bincount(bag,minlength=len(d))
    def observed(prob,tr):
        hp=s.math.hand_probabilities(prob,bag,len(d))[0];values=[]
        for ix in groups:
            if not tr[ix[0]]:continue
            e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])];post,ll,_=s.old.posterior(hp[ix],e)
            if post is not None:values.append(-ll/counts[ix].sum())
        return float(np.mean(values))
    for fold in range(4):
        tr=d['fold'].to_numpy()!=fold;p=s.math.initialize(s.old.initial(d,groups),bag)
        for step in range(1,7):
            rr,cat,w,used,minimum,excluded=s.targets(d,groups,bag,p,tr,aw)
            path=SOURCE/f'fold{fold}_em{step}.cbm';record=json.load(open(path.with_suffix('.json')))
            assert record['target_sha256']==s.old.checksum(rr,cat,w) and record['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            old=float(np.average(-np.log(p[rr,cat].clip(1e-300)),weights=w));old_nll=observed(p,tr)
            m=CatBoostClassifier();m.load_model(str(path));newp=m.predict_proba(x,thread_count=2)
            new=float(np.average(-np.log(newp[rr,cat].clip(1e-300)),weights=w));new_nll=observed(newp,tr)
            rows_out.append(dict(fold=fold,step=step,surrogate_before=old,surrogate_after=new,surrogate_change=new-old,observed_before=old_nll,observed_after=new_nll,observed_change=new_nll-old_nll,target_hash_exact=True))
            p=newp
        print('MSTEP_AUDIT',fold,flush=True)
    ROOT.mkdir(exist_ok=True);out=dict(steps=rows_out,surrogate_increases=sum(r['surrogate_change']>1e-10 for r in rows_out),observed_increases=sum(r['observed_change']>1e-10 for r in rows_out),
        limitations='All24 completed checkpoints of167, training-label objectives only. Weighted action CE is the fitted surrogate; observed list NLL is weighted1/actioncount per relationship to match the event objective. Numerical tiny-target pruning remains as in164. No model fitting, endpoint selection or leaderboard claim.')
    (ROOT/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
