\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from scipy.special import expit
from catboost import CatBoostClassifier
from session7_em import data
from session6_priority_model import load_models
from session7_likelihood import posterior,multilevel_inclusion
ROOT=Path('artifacts/evidence_session7');C=pl.col
def main():
    d=data().filter(C('behavior_family')=='soft_play').drop('row').with_row_index('row');models,cols=load_models('priority_ordered');X=d.select(cols).to_numpy();fv=d['fold'].to_numpy();ranks=d['evidence_rank'].fill_null(0).to_numpy();groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];out=[np.zeros((len(d),4)) for _ in range(2)];audit=[];start=time.time()
    for f in range(4):
        tr=fv!=f;va=fv==f;a=models['soft_play'][f][0].predict_proba(X,thread_count=4)[:,1];b=models['soft_play'][f][1].predict_proba(X,thread_count=4)[:,1];scale=np.maximum(1.00001,a+b);a/=scale;b/=scale
        check=expit(d['hu_check_r'].to_numpy()-d['partner_call_r'].to_numpy()).clip(.05,.95);prob=np.stack([1-a-b,a,b*(1-check),b*check],1)
        for iteration in range(2):
            target=np.zeros_like(prob);lls=[]
            for ix in groups:
                if not tr[ix[0]]:continue
                local=np.flatnonzero(ranks[ix]>0);e=local[np.argsort(ranks[ix][local])];post,ll,_=posterior(prob[ix],e);assert post is not None;target[ix]=post;lls.append(ll)
            row,category=np.nonzero((target>1e-7)&tr[:,None]);assert not va[row].any()
            m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='MultiClass',thread_count=4,random_seed=7100+11*f,verbose=False,allow_writing_files=False);m.fit(X[row],category,sample_weight=target[row,category]);prob=m.predict_proba(X,thread_count=4);out[iteration][va]=prob[va];m.save_model(str(ROOT/f'three_tier{iteration+1}_soft_play_fold{f}.cbm'));audit.append({'fold':f,'iteration':iteration+1,'mean_log_likelihood_before':float(np.mean(lls))})
        print('three tier fold',f,'seconds',round(time.time()-start,1),flush=True)
    other=pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet').join(pl.read_csv('data/development_labels.csv').select('pair_id','behavior_family'),on='pair_id').filter(C('behavior_family')!='soft_play').select('pair_id','hand_id','score')
    for it,p in enumerate(out,1):
        s=np.zeros(len(d))
        for ix in groups:s[ix]=multilevel_inclusion(p[ix])
        pred=d.select('pair_id','hand_id').with_columns(pl.Series('score',s));pl.concat([pred,other]).write_parquet(ROOT/f'three_tier{it}_oof.parquet')
        d.select('pair_id','hand_id').with_columns(*[pl.Series(f'p{k}',p[:,k]) for k in range(4)]).write_parquet(ROOT/f'three_tier{it}_probabilities.parquet')
    (ROOT/'three_tier_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
