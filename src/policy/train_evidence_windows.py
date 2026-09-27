\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,time,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
import build_outcome_roles as BOR, build_relationship_evidence as BRE
from evidence_data import load
root=Path('artifacts/policy'); OUT=Path('artifacts/evidence_windows'); OUT.mkdir(exist_ok=True); C=pl.col; NAMES=['directed_transfer','soft_play','coordinated_isolation']
relcols=json.loads((root/'relationship_evidence/columns.json').read_text()); folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
d,_=load(); hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
d=d.join(hands.select('hand_id','idx'),on='hand_id'); lab=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
W={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000),'w500_2500':(500,2500)}
cache=OUT/'window_features.parquet'
if cache.exists(): F=pl.read_parquet(cache)
else:
    parts=[]; t0=time.time()
    for w,(lo,hi) in W.items():
        z=d.filter((C('idx')>=lo)&(C('idx')<hi))
        for (table,),q in z.group_by('table_id'):
            if table not in tf: continue
            query=q.select('pair_id','hand_id').join(lab,on='pair_id')
            ro=BOR.build(table,query); re=BRE.build(table,query)
            qq=q.drop([c for c in q.columns if c.startswith(('outcome_','relationship_'))]).join(ro,on=['pair_id','hand_id']).join(re,on=['pair_id','hand_id'])
            parts.append(qq.select('pair_id','hand_id','table_id','behavior_family','evidence',*relcols).with_columns(pl.lit(w).alias('window')))
        print('features',w,round(time.time()-t0),flush=True)
    F=pl.concat(parts); F.write_parquet(cache)
def map5(df,col):
    r=[]
    for pid,g in df.sort(['pair_id',col,'hand_id'],descending=[False,True,False]).group_by('pair_id',maintain_order=True):
        if g['evidence'].sum()==0: continue
        e=g['evidence'].to_numpy()[:5]; r.append(float((np.cumsum(e)/np.arange(1,len(e)+1)*e).sum()/min(5,g['evidence'].sum())))
    return np.mean(r)
res={}
for mode in ['full_only','augmented']:
    parts=[]; t0=time.time()
    for f in range(4):
        va_tables={t for t,k in tf.items() if k==f}
        for n in NAMES:
            trw=['full'] if mode=='full_only' else list(W)
            tr=F.filter((C('behavior_family')==n)&C('window').is_in(trw)&~C('table_id').is_in(va_tables)); va=F.filter((C('behavior_family')==n)&C('table_id').is_in(va_tables))
            wt=np.where(tr['window'].to_numpy()=='full',1.0,0.5)
            m=CatBoostClassifier(iterations=850,depth=5,learning_rate=.035,loss_function='Logloss',l2_leaf_reg=8,thread_count=6,random_seed=710+f,verbose=False,allow_writing_files=False)
            m.fit(tr.select(relcols).to_numpy(),tr['evidence'].to_numpy(),sample_weight=wt)
            if mode=='augmented': m.save_model(str(OUT/f'{n}_fold{f}.cbm'))
            parts.append(va.select('pair_id','hand_id','window','behavior_family','evidence').with_columns(pl.Series('s',m.predict_proba(va.select(relcols).to_numpy())[:,1])))
        print(mode,'fold',f,round(time.time()-t0),flush=True)
    o=pl.concat(parts)
    for w in W: print(f"{mode:10s} {w:10s} MAP@5={map5(o.filter(C('window')==w),'s'):.4f}",' '.join(f"{b[:4]}={map5(o.filter((C('window')==w)&(C('behavior_family')==b)),'s'):.4f}" for b in NAMES),flush=True)
    if mode=='augmented': o.write_parquet(OUT/'oof.parquet')
