import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json
import numpy as np
import polars as pl
from sequence_features import augment

root=Path('artifacts/policy');dest=root/'joint_bags';dest.mkdir(exist_ok=True);C=pl.col
labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family')
h=pl.scan_parquet(str(root/'hand_features/*.parquet')).filter(C('phase')=='development').join(labs.select('pair_id').lazy(),on='pair_id').collect().sort('pair_id','time_index')
rc=[c for c in h.columns if c.endswith('_r')]
h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')]
h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz])
extra=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
d=pl.read_parquet('artifacts/dev_detail.parquet').join(labs,on='pair_id').join(h.select('pair_id','hand_id',*extra),on=['pair_id','hand_id']).with_columns((C('time')/.6).alias('relative_time'))
d,_=augment(d);d=d.join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0)).sort('pair_id','time','hand_id')
cols=json.loads((root/'sequence/evidence_columns.json').read_text());assert len(cols)==len(set(cols));X=d.select(cols).to_numpy().astype(np.float32);assert np.isfinite(X).all()
names=['none','directed_transfer','soft_play','coordinated_isolation'];bags=[];offset=0
for (pid,),q in d.group_by('pair_id',maintain_order=True):
    bags.append(dict(pair_id=pid,table_id=q['table_id'][0],start=offset,end=offset+len(q),label=int(q['label'][0]),behavior=names.index(q['behavior_family'][0])));offset+=len(q)
assert offset==len(d) and len(bags)==1860
np.save(dest/'features.npy',X);np.save(dest/'evidence.npy',d['evidence'].to_numpy().astype(np.float32));pl.DataFrame(bags).write_csv(dest/'bags.csv');d.select('pair_id','hand_id').write_parquet(dest/'hand_ids.parquet');(dest/'columns.json').write_text(json.dumps(cols))
print('saved',X.shape,'bags',len(bags),'evidence',d['evidence'].sum(),flush=True)
