\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
import sys
sys.path.insert(0, 'src/policy')
from pathlib import Path
import json, time
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from evidence_data import load

OUT=Path('artifacts/evidence_session4');OUT.mkdir(exist_ok=True)
root=Path('artifacts/policy')
config=dict(iterations=400,depth=5,learning_rate=.035,loss_function='Logloss',
            l2_leaf_reg=8,thread_count=4,verbose=False,allow_writing_files=False)
(OUT/'nested_config.json').write_text(json.dumps(config,indent=2))
d,_=load()
extra=pl.read_parquet(list((root/'outcome_roles').glob('T*.parquet'))).join(
    pl.read_parquet(list((root/'relationship_evidence').glob('T*.parquet'))),
    on=['pair_id','hand_id'],validate='1:1')
d=d.join(extra,on=['pair_id','hand_id'],validate='1:1').sort('pair_id','hand_id')
cols=json.loads((root/'relationship_evidence/columns.json').read_text())
folds=json.loads(Path('artifacts/folds.json').read_text())
tf={t:f['fold'] for f in folds for t in f['valid_tables']}
fv=np.array([tf[t] for t in d['table_id']]);X=d.select(cols).to_numpy();y=d['evidence'].to_numpy()
families=d['behavior_family'].to_numpy()
assert np.isfinite(X).all()
d.select('pair_id','hand_id','table_id','behavior_family','evidence','time','net_direction').with_columns(pl.Series('fold',fv)).write_parquet(OUT/'hand_index.parquet')
t=time.time()
for outer in range(4):
    path=OUT/f'nested_outer{outer}.parquet'
    if path.exists():continue
    scores=np.full(len(d),np.nan)
    for family in ['directed_transfer','soft_play','coordinated_isolation']:
        fm=families==family
                                                                                         
        for valfold in range(4):
            tr=fm&(fv!=outer)&(fv!=valfold)
            va=fm&(fv==valfold)
            assert not np.any(tr & (fv==outer))
            m=CatBoostClassifier(**config,random_seed=1710+outer*10+valfold)
            m.fit(X[tr],y[tr])
            scores[va]=m.predict_proba(X[va])[:,1]
            if valfold==outer:m.save_model(str(OUT/f'base_{family}_fold{outer}.cbm'))
        print('outer',outer,family,'elapsed',round(time.time()-t,1),flush=True)
    assert np.isfinite(scores).all()
    d.select('pair_id','hand_id').with_columns(pl.Series('base_score',scores)).write_parquet(path)
print('nested cross-fit complete',round(time.time()-t,1),flush=True)
