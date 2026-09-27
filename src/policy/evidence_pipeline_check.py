\
\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy'); sys.path.insert(0,'artifacts/reference_metric')
import json,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sequence_features import augment
import build_outcome_roles as BOR, build_relationship_evidence as BRE
C=pl.col; root=Path('artifacts/policy'); NAMES=['directed_transfer','soft_play','coordinated_isolation']
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
relcols=json.loads((root/'relationship_evidence/columns.json').read_text())
models={n:[] for n in NAMES}
for n in NAMES:
    for f in range(4): m=CatBoostClassifier(); m.load_model(str(root/f'relationship_evidence/{n}_fold{f}.cbm')); models[n].append(m)
lab=pl.read_csv('data/development_labels.csv'); pos=lab.filter(C('label')==1).select('pair_id','player_1','player_2','behavior_family')
sel=[]
for path in sorted(Path('artifacts/detail_features').glob('*.parquet')):
    table=path.stem
    if table not in tf: continue
    d=pl.read_parquet(path).filter(C('phase')=='development').join(pos.select('pair_id',C('behavior_family').alias('family')),on='pair_id')
    if d.is_empty(): continue
    d=d.with_columns((C('time')/.6).alias('relative_time'))
    h=pl.read_parquet(root/'hand_features'/path.name).filter(C('phase')=='development').join(pos.select('pair_id'),on='pair_id').sort('pair_id','time_index'); rcols=[c for c in h.columns if c.endswith('_r')]
    h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rcols]); hz=[c for c in h.columns if c.endswith('_hz')]
    h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]); add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
    d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']); d,_=augment(d)
    query=d.select('pair_id','hand_id').join(pos.select('pair_id','player_1','player_2'),on='pair_id')
    ro=BOR.build(table,query); re=BRE.build(table,query); d=d.join(ro,on=['pair_id','hand_id']).join(re,on=['pair_id','hand_id'])
    scores=[]
    for n in NAMES:
        z=d.filter(C('family')==n)
        if z.is_empty(): continue
        s=models[n][tf[table]].predict_proba(z.select(relcols).to_numpy(),thread_count=6)[:,1]
        scores.append(z.select('pair_id','hand_id').with_columns(pl.Series('score',s)))
    sel.append(pl.concat(scores))
o=pl.concat(sel).join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0))
print('pairs',o['pair_id'].n_unique(),'hands',o.height,'evidence hands matched',int(o['evidence'].sum()))
r=[]
for pid,g in o.sort(['pair_id','score','hand_id'],descending=[False,True,False]).group_by('pair_id',maintain_order=True):
    e=g['evidence'].to_numpy()[:5]; r.append(float((np.cumsum(e)/np.arange(1,len(e)+1)*e).sum()/min(5,g['evidence'].sum())))
print('MAP@5 via assembly code path on dev:',round(np.mean(r),4))
                                                  
oo=pl.read_parquet(root/'relationship_evidence/oof.parquet')
r2=[]
for pid,g in oo.sort(['pair_id','score','hand_id'],descending=[False,True,False]).group_by('pair_id',maintain_order=True):
    e=g['evidence'].to_numpy()[:5]; r2.append(float((np.cumsum(e)/np.arange(1,len(e)+1)*e).sum()/min(5,g['evidence'].sum())))
print('MAP@5 stored OOF:',round(np.mean(r2),4))
                          
j=o.join(oo.select('pair_id','hand_id',C('score').alias('oof')),on=['pair_id','hand_id']); print('score corr assembly vs OOF:',round(float(np.corrcoef(j['score'],j['oof'])[0,1]),4),'max abs diff',round(float((j['score']-j['oof']).abs().max()),4))
