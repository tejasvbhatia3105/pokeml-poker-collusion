import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4');sys.path.insert(0,'src/policy')
from pathlib import Path
import json,time,shutil
import numpy as np,polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from evidence_data import load
ROOT=Path('artifacts/evidence_session4');OUT=ROOT/'nested_blend';OUT.mkdir(exist_ok=True)
shutil.copyfile(ROOT/'hand_index.parquet',OUT/'hand_index.parquet')
r=Path('artifacts/policy');d,_=load();extra=pl.read_parquet(list((r/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((r/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
d=d.join(extra,on=['pair_id','hand_id']).join(pl.read_parquet(ROOT/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
cols=json.loads((r/'relationship_evidence/columns.json').read_text());X=d.select(cols).to_numpy();y=d['evidence'].to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy()
config=json.loads((ROOT/'leafwise_config.json').read_text());t=time.time()
outer_oof=pl.read_parquet(ROOT/'leafwise_separate_oof.parquet').select('pair_id','hand_id',pl.col('score').alias('leaf'))
with threadpool_limits(limits=4):
    for outer in range(4):
        path=OUT/f'nested_outer{outer}.parquet'
        if path.exists():continue
        scores=np.full(len(d),np.nan)
        for val in range(4):
            if val==outer:continue
            for b in ['directed_transfer','soft_play','coordinated_isolation']:
                tr=(fv!=outer)&(fv!=val)&(fam==b);va=(fv==val)&(fam==b)
                m=HistGradientBoostingClassifier(**config,random_state=4710+outer*10+val,categorical_features=None)
                m.fit(X[tr],y[tr]);scores[va]=m.predict_proba(X[va])[:,1]
        z=d.select('pair_id','hand_id','fold').with_columns(pl.Series('inner_leaf',scores)).join(outer_oof,on=['pair_id','hand_id']).join(pl.read_parquet(ROOT/f'nested_outer{outer}.parquet'),on=['pair_id','hand_id'])
        z=z.with_columns(((pl.col('base_score')+pl.when(pl.col('fold')==outer).then(pl.col('leaf')).otherwise(pl.col('inner_leaf')))/2).alias('base_score'))
        assert np.isfinite(z['base_score'].to_numpy()).all();z.select('pair_id','hand_id','base_score').write_parquet(path)
        print('nested blend outer',outer,'seconds',round(time.time()-t,1),flush=True)
