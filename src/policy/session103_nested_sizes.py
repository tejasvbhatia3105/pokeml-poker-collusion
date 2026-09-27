\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import itertools,json,time,hashlib
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from session22_size_density import labels
ROOT=Path('artifacts/evidence_session103_nested_sizes')
C=pl.col

def train():
    ROOT.mkdir(exist_ok=True);paths=sorted(Path('artifacts/evidence_session22_size_density/samples').glob('T*.parquet'));assert len(paths)==400
    d=pl.read_parquet(paths);pc=json.load(open('artifacts/policy/feature_columns.json'));fv=d['fold'].to_numpy();x=d.select(pc).to_numpy();size=d['log_bet_ratio'].to_numpy()
    config={'method':__doc__,'model':'400D6 lr.05 L2=15 MultiClass','sample':'original22 250 raises/table seed2222',
            'bins':'up to16 non-allin training quantiles plus allin atom','seed':'22220+4*f+h sorted folds',
            'training':'development gameplay only, no evidence labels','action_reference':'84/action_exclude{f}{h}.cbm'}
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));records=[];start=time.time()
    for f,h in itertools.combinations(range(4),2):
        tr=(fv!=f)&(fv!=h);va=~tr;v=size[tr&~d['allin'].to_numpy()]
        edges=np.unique(np.quantile(v,np.linspace(0,1,17)));edges[0]=min(-8.,float(v.min())-.1);edges[-1]=max(8.,float(v.max())+.1)
        y=labels(d,edges);n=len(edges)-1;assert set(y[tr])==set(range(n+1))
        centers=np.array([size[tr&(y==k)].mean() for k in range(n+1)])
        path=ROOT/f'size_exclude{f}{h}.cbm'
        m=CatBoostClassifier(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=15,loss_function='MultiClass',random_seed=22220+4*f+h,thread_count=2,allow_writing_files=False,verbose=False)
        m.fit(x[tr],y[tr]);m.save_model(str(path))
        check=CatBoostClassifier();check.load_model(str(path))
        p=m.predict_proba(x[va],thread_count=2);np.testing.assert_array_equal(p,check.predict_proba(x[va],thread_count=2))
        prior=np.bincount(y[tr],minlength=n+1)/tr.sum()
        record={'excluded_folds':[f,h],'training_rows':int(tr.sum()),'query_rows':int(va.sum()),
                'training_tables':sorted(d.filter(pl.Series(tr))['table_id'].unique().to_list()),
                'validation_overlap':0,'edges':edges.tolist(),'centers':centers.tolist(),'classes':n+1,
                'conditional_logloss':float(-np.log(p[np.arange(va.sum()),y[va]].clip(1e-12)).mean()),
                'prior_logloss':float(-np.log(prior[y[va]].clip(1e-12)).mean()),
                'model_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'replay_error':0,
                'seconds':time.time()-start}
        (ROOT/f'metadata_exclude{f}{h}.json').write_text(json.dumps(record,indent=2));records.append(record)
        (ROOT/'audit.json').write_text(json.dumps(records,indent=2));print('SIZE_DONE',f,h,record['conditional_logloss'],record['prior_logloss'],record['seconds'],flush=True)

if __name__=='__main__':train()
