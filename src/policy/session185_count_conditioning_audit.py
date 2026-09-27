\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,itertools,hashlib
from pathlib import Path
import numpy as np,polars as pl
from scipy.special import softmax
from catboost import CatBoostClassifier
import session123_family_holdout as s
ROOT=Path('artifacts/evidence_session185_count_conditioning_audit')

def tail_gradient(p,k):
    q=p[:,1:].sum(1);n=len(q);f=np.zeros((n+1,k+1));b=np.zeros((n+1,k));f[0,0]=b[n,0]=1
    for i in range(n):
        f[i+1,:k]=f[i,:k]*(1-q[i]);f[i+1,1:k]+=f[i,:k-1]*q[i]
        f[i+1,k]=f[i,k]+f[i,k-1]*q[i]
    for i in range(n-1,-1,-1):
        b[i]=b[i+1]*(1-q[i]);b[i,1:]+=b[i+1,:-1]*q[i]
    z=f[n,k];assert z>0
    mass=np.sum(f[:n,:k]*b[1:,::-1],axis=1)
    gradient=p[:,1:]*p[:,:1]*(mass/z)[:,None]
    return z,gradient

def check():
    rng=np.random.default_rng(18501);p=rng.dirichlet([3,1,1],6);z=np.log(p[:,1:]/p[:,:1]);errors=[];graderrors=[]
    for k in [1,3,5]:
        prob,g=tail_gradient(p,k);exact=0.
        for active in itertools.product([False,True],repeat=len(p)):
            if sum(active)>=k:exact+=np.prod(np.where(active,p[:,1:].sum(1),p[:,0]))
        errors.append(abs(prob-exact))
        for i in range(len(p)):
            for j in range(2):
                v=z.copy();v[i,j]+=1e-6;a=tail_gradient(softmax(np.column_stack([np.zeros(len(p)),v]),axis=1),k)[0]
                v[i,j]-=2e-6;b=tail_gradient(softmax(np.column_stack([np.zeros(len(p)),v]),axis=1),k)[0]
                graderrors.append(abs(g[i,j]-(np.log(a)-np.log(b))/2e-6))
    assert max(errors)<1e-12 and max(graderrors)<1e-8
    return dict(exhaustive_binary_paths=3*2**6,logit_gradient_checks=len(graderrors),tail_error=max(errors),gradient_error=max(graderrors))

def main():
    ROOT.mkdir(exist_ok=True);proof=check();v=json.load(open(s.ROOT/'verification.json'));assert v['models']==32 and v['final_prediction_error']==0
    d,x,groups=s.load();ranks=d['evidence_rank'].fill_null(0).to_numpy();rows=[];models=[]
    for fold in range(4):
        path=s.ROOT/'all'/f'fold{fold}_em2.cbm';record=json.load(open(path.with_suffix('.json')));assert record['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        m=CatBoostClassifier();m.load_model(str(path));p=m.predict_proba(x,thread_count=2);k=record['minimum']
        for ix in groups:
            e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])];post,ll,_=s.posterior(p[ix],e)
            if post is None:continue
            mass,g=tail_gradient(p[ix],k);ug=p[ix,1:]-post[:,1:];u=float(abs(ug).sum());extra=float(abs(g).sum())
            rows.append(dict(fit_fold=fold,pair_id=d['pair_id'][int(ix[0])],table_id=d['table_id'][int(ix[0])],family=d['behavior_family'][int(ix[0])],
                split='heldout' if d['fold'][int(ix[0])]==fold else 'train',tail_probability=mass,expected_events=float(p[ix,1:].sum()),
                unconditional_NLL_per_truth=-ll/len(e),conditional_NLL_per_truth=(-ll+np.log(mass))/len(e),
                unconditional_gradient_L1=u,count_gradient_L1=extra,count_to_unconditional_gradient_L1=extra/max(u,1e-15),gradient_sign_changes=int(((ug+g)*ug<0).sum())))
        models.append(dict(fold=fold,minimum=k,model_sha256=record['model_sha256']))
    q=pl.DataFrame(rows);q.write_csv(ROOT/'pair_terms.csv');summary=[]
    for (split,),g in q.group_by('split'):
        summary.append(dict(split=split,pair_predictions=len(g),tail_quantiles=np.quantile(g['tail_probability'],[0,.1,.5,.9,1]).tolist(),
            tail_below_50=int((g['tail_probability']<.5).sum()),tail_below_90=int((g['tail_probability']<.9).sum()),
            mean_NLL_change=float((g['conditional_NLL_per_truth']-g['unconditional_NLL_per_truth']).mean()),
            aggregate_gradient_L1_ratio=g['count_gradient_L1'].sum()/g['unconditional_gradient_L1'].sum(),
            per_pair_gradient_ratio_quantiles=np.quantile(g['count_to_unconditional_gradient_L1'],[.1,.5,.9]).tolist(),gradient_sign_changes=int(g['gradient_sign_changes'].sum())))
    result=dict(math_checks=proof,summary=summary,models=models,limitations=__doc__)
    (ROOT/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
