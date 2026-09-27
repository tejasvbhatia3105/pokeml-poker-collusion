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
from session8_data import hand_data,targets,ROOT,C
from session6_priority import inclusion
def hand_probability(p,g,n):
    return -np.expm1(np.bincount(g,weights=np.log1p(-np.clip(p,1e-9,1-1e-9)),minlength=n))
def main():
    d=hand_data();context=os.environ.get('WITNESS_CONTEXT','0')=='1';prefix='witness_context' if context else 'witness';a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet');acols=[c for c in a.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family','time']]
    hc=[c for c in d.columns if c.startswith('relationship_')]+[c for c in ['fold_partner','call_partner','check_hu','both_showdown','relative_time','strong_net','weak_net'] if c in d.columns]
    a=a.select('pair_id','hand_id',*acols).join(d.select('pair_id','hand_id',C('row').alias('hand_row'),*hc),on=['pair_id','hand_id'],validate='m:1',maintain_order='left');cols=acols+hc
    if context:
        extra=pl.read_parquet(ROOT/'neighbor_actions.parquet');extra_cols=[c for c in extra.columns if c.startswith('neighbor_')];a=a.join(extra,on=['pair_id','hand_id','action_no'],validate='1:1',maintain_order='left');cols+=extra_cols
    X=a.select(cols).to_numpy().astype('float32');assert np.isfinite(X).all();g=a['hand_row'].to_numpy();n=len(d);counts=np.bincount(g,minlength=n);assert counts.min()>0
    predictions=[np.zeros((n,2)) for _ in range(2)];start=time.time();audit=[]
    for f in range(4):
        for b in ['directed_transfer','soft_play','coordinated_isolation']:
            t1,t2,e1,e2,va=targets(d,f,b);av=va[g]
            for k,truth,eligible in [(1,t1,e1),(2,t2,e2)]:
                tr=eligible[g];gt=g[tr];yt=truth[gt];resp=yt/counts[gt]
                for it in range(2):
                    m=CatBoostClassifier(iterations=350,depth=5,learning_rate=.04,l2_leaf_reg=8,loss_function='CrossEntropy',thread_count=4,random_seed=8110+11*f+k,verbose=False,allow_writing_files=False);m.fit(X[tr],resp)
                    pv=m.predict_proba(X[av],thread_count=4)[:,1];predictions[it][va,k-1]=hand_probability(pv,g[av],n)[va];m.save_model(str(ROOT/f'{prefix}{it+1}_event{k}_{b}_fold{f}.cbm'))
                    p=m.predict_proba(X[tr],thread_count=4)[:,1];q=hand_probability(p,gt,n);resp=np.where(yt,p/np.maximum(q[gt],1e-9),0);assert resp.min()>=0 and resp.max()<=1+1e-8
                    audit.append({'fold':f,'family':b,'head':k,'iteration':it+1,'training_actions':int(tr.sum()),'training_hands':int(eligible.sum()),'mean_positive_witness_count':float(np.bincount(gt,weights=resp,minlength=n)[truth&eligible].mean())})
            print(prefix,'fold',f,b,'seconds',round(time.time()-start,1),flush=True)
    for it,p in enumerate(predictions,1):
        q=d.select('pair_id','hand_id','time').with_columns(pl.Series('primary',p[:,0]),pl.Series('secondary',p[:,1]));parts=[]
        for _,z in q.group_by('pair_id',maintain_order=True):parts.append(z.with_columns(pl.Series('score',inclusion(z['primary'].to_numpy(),z['secondary'].to_numpy()))))
        pl.concat(parts).write_parquet(ROOT/f'{prefix}{it}_oof.parquet')
    (ROOT/f'{prefix}_columns.json').write_text(json.dumps(cols));(ROOT/f'{prefix}_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
