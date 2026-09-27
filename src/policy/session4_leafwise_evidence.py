\
\
\
\
import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
sys.path.insert(0,'src/policy')
import json,time,joblib
from pathlib import Path
import numpy as np,polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from evidence_data import load
OUT=Path('artifacts/evidence_session4');root=Path('artifacts/policy');C=pl.col
N=['directed_transfer','soft_play','coordinated_isolation']
d,_=load();extra=pl.read_parquet(list((root/'outcome_roles').glob('T*.parquet'))).join(
    pl.read_parquet(list((root/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
d=d.join(extra,on=['pair_id','hand_id']).join(pl.read_parquet(OUT/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
cols=json.loads((root/'relationship_evidence/columns.json').read_text());X=d.select(cols).to_numpy();y=d['evidence'].to_numpy();fv=d['fold'].to_numpy()
family=np.array([N.index(n) for n in d['behavior_family']]);config=dict(max_iter=300,learning_rate=.05,max_leaf_nodes=15,
    min_samples_leaf=20,l2_regularization=10.,max_bins=127,early_stopping=False)
(OUT/'leafwise_config.json').write_text(json.dumps(config,indent=2))
t=time.time();out=d.select('pair_id','hand_id','behavior_family','evidence','fold')
with threadpool_limits(limits=4):
    for mode in ['separate','pooled']:
        path=OUT/f'leafwise_{mode}_oof.parquet'
        if path.exists():continue
        pr=np.full(len(d),np.nan)
        for f in range(4):
            groups=range(3) if mode=='separate' else [-1]
            xx=X if mode=='separate' else np.column_stack([X,family])
            for group in groups:
                fm=family==group if group>=0 else np.ones(len(d),bool)
                tr=(fv!=f)&fm;va=(fv==f)&fm
                cat=None if mode=='separate' else [False]*len(cols)+[True]
                m=HistGradientBoostingClassifier(**config,random_state=4710+f,categorical_features=cat)
                m.fit(xx[tr],y[tr]);pr[va]=m.predict_proba(xx[va])[:,1]
                joblib.dump(m,OUT/f'leafwise_{mode}_{group}_fold{f}.joblib',compress=3)
            print(mode,'fold',f,round(time.time()-t,1),flush=True)
        assert np.isfinite(pr).all()
        out.with_columns(pl.Series('score',pr)).write_parquet(path)
