\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from sequence_features import augment
from session8_data import hand_data,targets,ROOT,C
from session6_priority import inclusion
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
def data():
    full=hand_data();cols=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['event']
    ix=full.select('pair_id','hand_id','time','fold','net_direction','row');w=pl.read_parquet('artifacts/evidence_windows/window_features.parquet').join(ix,on=['pair_id','hand_id'],validate='m:1',maintain_order='left');parts=[full.with_columns(pl.lit('full').alias('window'))]
    for name,lo,hi in [('first_2000',0,2000),('last_2000',1000,3000)]:
        q=w.filter(C('window')==name).with_columns(pl.lit('development').alias('phase'),((C('time')*5000-lo)/(hi-lo)).alias('relative_time'));q,_=augment(q);parts.append(q)
    keep=['pair_id','hand_id','time','fold','row','window','behavior_family','evidence']+cols
    keep=list(dict.fromkeys(keep));d=pl.concat([p.select(keep) for p in parts],how='vertical_relaxed')
    assert len(d)==sum(map(len,parts)) and d.select('pair_id','hand_id','window').is_duplicated().sum()==0
    return full,d,cols
def main():
    full,d,cols=data();X=d.select(cols).to_numpy();mapping=d['row'].to_numpy();fv=d['fold'].to_numpy();family=d['behavior_family'].to_numpy();weight=np.where(d['window'].to_numpy()=='full',1.,.5);pred={name:np.zeros((len(d),2)) for name in ['exposure_cat','exposure_hist']};start=time.time();audit=[]
    with threadpool_limits(limits=4):
        for f in range(4):
            for b in FAMILIES:
                ts=targets(full,f,b)
                for k,y,tr in [(1,ts[0],ts[2]),(2,ts[1],ts[3])]:
                    use=tr[mapping];va=(fv==f)&(family==b);assert not np.any(use&va);assert np.all(fv[use]!=f)
                    audit.append({'fold':f,'family':b,'event':k,'eligible_original':int(tr.sum()),'eligible_augmented':int(use.sum()),'weight_sum':float(weight[use].sum())})
                    models={'exposure_cat':CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False),'exposure_hist':HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+f)}
                    for name,m in models.items():
                        m.fit(X[use],y[mapping][use],sample_weight=weight[use])
                        pred[name][va,k-1]=m.predict_proba(X[va])[:,1]
                        if name.endswith('cat'):m.save_model(str(ROOT/f'{name}_event{k}_{b}_fold{f}.cbm'))
                        else:joblib.dump(m,ROOT/f'{name}_event{k}_{b}_fold{f}.joblib')
                print('exposure',f,b,'seconds',round(time.time()-start,1),flush=True)
    for name,p in pred.items():
        assert np.isfinite(p).all();q=d.select('pair_id','hand_id','time','window').with_columns(pl.Series('primary',p[:,0]),pl.Series('secondary',p[:,1]));parts=[]
        for _,g in q.group_by('window','pair_id',maintain_order=True):
            g=g.sort('time','hand_id');parts.append(g.with_columns(pl.Series('score',inclusion(g['primary'].to_numpy(),g['secondary'].to_numpy()))))
        z=pl.concat(parts);z.filter(C('window')=='full').drop('window').write_parquet(ROOT/f'{name}_oof.parquet');z.filter(C('window')!='full').write_parquet(ROOT/f'{name}_window_predictions.parquet')
    (ROOT/'exposure_columns.json').write_text(json.dumps(cols,indent=2));(ROOT/'exposure_training_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
