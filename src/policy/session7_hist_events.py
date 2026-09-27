\
\
\
\
import os,json,time,joblib
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session7_em import data
from session6_priority import training_targets,inclusion
ROOT=Path('artifacts/evidence_session7');OLD=Path('artifacts/evidence_session6');C=pl.col
def main():
    d=data().join(pl.read_parquet(OLD/'order_subtype_labels.parquet'),on=['pair_id','hand_id'],how='left',maintain_order='left',validate='1:1').with_columns(C('subtype').fill_null(0));config=json.loads((OLD/'priority_ordered_columns.json').read_text());cols=config['event'];tc=config['type'];X=d.select(cols).to_numpy();XT=d.select(tc).to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();sub=d['subtype'].to_numpy();y=d['evidence'].to_numpy();rank=d['evidence_rank'].fill_null(0).to_numpy();t=d['time'].to_numpy();groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];pred=np.zeros((len(d),2));start=time.time()
    with threadpool_limits(limits=4):
        for f in range(4):
            for b in ['directed_transfer','soft_play','coordinated_isolation']:
                tr=(fv!=f)&(fam==b);va=(fv==f)&(fam==b);typ=CatBoostClassifier();typ.load_model(str(OLD/f'priority_ordered_type_{b}_fold{f}.cbm'));tp=typ.predict_proba(XT,thread_count=4)[:,1];a,bt,ea,eb=training_targets(y,sub,tp,tr,t,groups,rank)
                for k,target,eligible in [(1,a,ea),(2,bt,eb)]:
                    assert not np.any(eligible&va)
                    m=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+f);m.fit(X[eligible],target[eligible]);pred[va,k-1]=m.predict_proba(X[va])[:,1];joblib.dump(m,ROOT/f'hist_event{k}_{b}_fold{f}.joblib',compress=3)
            print('hist events fold',f,'seconds',round(time.time()-start,1),flush=True)
    q=d.select('pair_id','hand_id','time').with_columns(pl.Series('primary',pred[:,0]),pl.Series('secondary',pred[:,1]));old=pl.read_parquet(OLD/'priority_ordered_oof.parquet').select('pair_id','hand_id',C('primary').alias('cat_primary'),C('secondary').alias('cat_secondary'));q=q.join(old,on=['pair_id','hand_id'],validate='1:1');parts=[];blended=[]
    for _,g in q.group_by('pair_id'):
        g=g.sort('time','hand_id');a=g['primary'].to_numpy();b=g['secondary'].to_numpy();ca=g['cat_primary'].to_numpy();cb=g['cat_secondary'].to_numpy();s=np.maximum(1,a+b);cs=np.maximum(1,ca+cb)
        parts.append(g.select('pair_id','hand_id','time','primary','secondary').with_columns(pl.Series('score',inclusion(a,b))));blended.append(g.select('pair_id','hand_id').with_columns(pl.Series('score',inclusion(.5*a/s+.5*ca/cs,.5*b/s+.5*cb/cs))))
    pl.concat(parts).write_parquet(ROOT/'hist_events_oof.parquet');pl.concat(blended).write_parquet(ROOT/'hist_eventblend_oof.parquet')
if __name__=='__main__':main()
