\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from evidence_data import load
from session6_priority_model import load_models
from session6_priority import inclusion
from session7_likelihood import posterior
ROOT=Path('artifacts/evidence_session7');C=pl.col
def data():
    d,_=load();p=Path('artifacts/policy')
    extra=pl.read_parquet(list((p/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((p/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(extra,on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'],validate='1:1')
    return d.join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id','evidence_rank'),on=['pair_id','hand_id'],how='left',validate='1:1').sort('pair_id','time','hand_id').with_row_index('row')
def main():
    d=data();models,cols=load_models('priority_ordered');X=d.select(cols).to_numpy();fv=d['fold'].to_numpy();family=d['behavior_family'].to_numpy();ranks=d['evidence_rank'].fill_null(0).to_numpy()
    groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)]
    outputs=[np.full((len(d),3),np.nan) for _ in range(2)];audit=[];start=time.time()
    for f in range(4):
        for b in models:
            tr=(fv!=f)&(family==b);va=(fv==f)&(family==b);p1=models[b][f][0].predict_proba(X,thread_count=4)[:,1];p2=models[b][f][1].predict_proba(X,thread_count=4)[:,1];scale=np.maximum(1.00001,p1+p2);prob=np.stack([1-(p1+p2)/scale,p1/scale,p2/scale],1)
            for iteration in range(2):
                target=np.zeros_like(prob);used=np.zeros(len(d),bool);lls=[];ent=[]
                for ix in groups:
                    if not tr[ix[0]]:continue
                    local=np.flatnonzero(ranks[ix]>0);e=local[np.argsort(ranks[ix][local])]
                    post,ll,weights=posterior(prob[ix],e)
                    if post is None:continue
                    target[ix]=post;used[ix]=True;lls.append(ll);ent.append(-sum(w*np.log(w+1e-15) for w in weights.values()))
                assert not np.any(used&va)
                                                                             
                row,category=np.nonzero((target>1e-7)&used[:,None]);weight=target[row,category]
                model=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='MultiClass',thread_count=4,random_seed=7100+11*f,verbose=False,allow_writing_files=False)
                model.fit(X[row],category,sample_weight=weight);prob=model.predict_proba(X,thread_count=4)
                outputs[iteration][va]=prob[va];model.save_model(str(ROOT/f'em{iteration+1}_{b}_fold{f}.cbm'))
                audit.append({'fold':f,'family':b,'iteration':iteration+1,'training_hands':int(used.sum()),'weighted_rows':len(row),'mean_training_log_likelihood_before':float(np.mean(lls)),'mean_split_entropy':float(np.mean(ent))})
            print('EM fold',f,b,'seconds',round(time.time()-start,1),flush=True)
    for iteration,p in enumerate(outputs,1):
        assert np.isfinite(p).all();parts=[];q=d.select('pair_id','hand_id','time').with_columns(pl.Series('primary',p[:,1]),pl.Series('secondary',p[:,2]))
        for _,g in q.group_by('pair_id',maintain_order=True):parts.append(g.with_columns(pl.Series('score',inclusion(g['primary'].to_numpy(),g['secondary'].to_numpy()))))
        pl.concat(parts).write_parquet(ROOT/f'em{iteration}_oof.parquet')
    (ROOT/'em_columns.json').write_text(json.dumps(cols));(ROOT/'em_training_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
