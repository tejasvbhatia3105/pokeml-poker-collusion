\
\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sequence_features import augment
import build_outcome_roles as BOR, build_relationship_evidence as BRE
from evidence_data import load
root=Path('artifacts/policy'); C=pl.col; NAMES=['directed_transfer','soft_play','coordinated_isolation']
relcols=json.loads((root/'relationship_evidence/columns.json').read_text()); folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
models={n:[] for n in NAMES}
for n in NAMES:
    for f in range(4): m=CatBoostClassifier(); m.load_model(str(root/f'relationship_evidence/{n}_fold{f}.cbm')); models[n].append(m)
d,_=load()                                                                                   
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
d=d.join(hands.select('hand_id','idx'),on='hand_id'); lab=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
def map5(df,col):
    r=[]
    for pid,g in df.sort(['pair_id',col,'hand_id'],descending=[False,True,False]).group_by('pair_id',maintain_order=True):
        e=g['evidence'].to_numpy()[:5]
        if g['evidence'].sum()==0: continue
        r.append(float((np.cumsum(e)/np.arange(1,len(e)+1)*e).sum()/min(5,g['evidence'].sum())))
    return np.mean(r),len(r)
for w,(lo,hi) in {'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}.items():
    z=d.filter((C('idx')>=lo)&(C('idx')<hi)); parts=[]
    for (table,),q in z.group_by('table_id'):
        if table not in tf: continue
        query=q.select('pair_id','hand_id').join(lab,on='pair_id')
        ro=BOR.build(table,query); re=BRE.build(table,query)
        qq=q.drop([c for c in q.columns if c.startswith(('outcome_','relationship_'))]).join(ro,on=['pair_id','hand_id']).join(re,on=['pair_id','hand_id'])
        sc=np.zeros(qq.height)
        for n in NAMES:
            m=(qq['behavior_family']==n).to_numpy()
            if m.any(): sc[m]=models[n][tf[table]].predict_proba(qq.filter(pl.Series(m)).select(relcols).to_numpy(),thread_count=6)[:,1]
        parts.append(qq.select('pair_id','hand_id','behavior_family','evidence').with_columns(pl.Series('s',sc)))
    o=pl.concat(parts); m,npairs=map5(o,'s')
    print(f"{w:10s} pairs with evidence in window={npairs} hands={o.height} MAP@5 (window-recomputed features) = {m:.4f}",' '.join(f"{b[:4]}={map5(o.filter(C('behavior_family')==b),'s')[0]:.4f}" for b in NAMES),flush=True)
