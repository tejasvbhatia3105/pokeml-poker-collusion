\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,time,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sequence_features import augment
root=Path('artifacts/policy'); C=pl.col; out=Path(sys.argv[1]); out.mkdir(exist_ok=True,parents=True)
S='cache/'
seqcols=json.loads((root/'sequence/evidence_columns.json').read_text())
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
lab=pl.read_csv('data/development_labels.csv')
def hand_matrix(table,phase,pairs):
    h=pl.read_parquet(root/'hand_features'/f'{table}.parquet').filter(C('phase')==phase).join(pairs,on='pair_id').sort('pair_id','time_index'); rc=[c for c in h.columns if c.endswith('_r')]
    if h.is_empty(): return None
    h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]); hz=[c for c in h.columns if c.endswith('_hz')]
    h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]); add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
    det=pl.read_parquet(f'artifacts/detail_features/{table}.parquet').filter(C('phase')==phase) if phase=='evaluation' or True else None
    if phase=='development':
                                                                                  
        dd=DEVDETAIL.filter(C('table_id')==table); det=pl.concat([det,dd.select(det.columns)]).unique(subset=['pair_id','hand_id'])
    d=det.join(pairs,on='pair_id').with_columns(((C('time')/.6) if phase=='development' else ((C('time')-.6)/.4)).alias('relative_time')).join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id'])
    if d.is_empty(): return None
    d,_=augment(d); return d
DEVDETAIL=pl.read_parquet('artifacts/dev_detail.parquet')
                                                                                                 
t0=time.time(); models=[]
tr_parts=[]
for table,q in DEVDETAIL.group_by('table_id'):
    table=table[0]; d=hand_matrix(table,'development',lab.select('pair_id','label'))
    if d is None: continue
    d=d.join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0))
    tr_parts.append(d.filter((C('evidence')==1)|(C('label')==0)).select('pair_id','table_id','evidence',*seqcols))
tr=pl.concat(tr_parts); X=tr.select(seqcols).to_numpy(); y=tr['evidence'].to_numpy(); g=np.array([tf.get(t,-1) for t in tr['table_id']])
print('training rows',len(tr),'pos',int(y.sum()),round(time.time()-t0),flush=True)
for f in range(4):
    m=CatBoostClassifier(iterations=800,depth=6,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=8,thread_count=6,random_seed=5+f,verbose=False,allow_writing_files=False,scale_pos_weight=float((y[g!=f]==0).sum()/max(1,(y[g!=f]==1).sum())))
    m.fit(X[g!=f],y[g!=f]); m.save_model(str(out/f'hand_fold{f}.cbm')); models.append(m); print('fold',f,round(time.time()-t0),flush=True)
                                                                                                              
ev=pl.read_csv('data/evaluation_pairs.csv').select('pair_id'); known=pl.concat([lab.select('pair_id'),ev]).unique()
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
WINDOWS={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000),'w500_2500':(500,2500),'w0_1500':(0,1500),'w750_2250':(750,2250),'w1500_3000':(1500,3000),'evaluation':(3000,5000)}
def agg(z,name):
    c=C('hs'); return z.group_by('pair_id').agg(pl.len().alias('hs_n'),c.max().alias('hs_max'),c.top_k(3).mean().alias('hs_top3'),c.top_k(5).mean().alias('hs_top5'),(c>.5).sum().alias('hs_c50'),(c>.8).sum().alias('hs_c80'),c.mean().alias('hs_mean'),(c/(1-c+1e-6)).log().clip(-6,6).top_k(5).sum().alias('hs_lr5')).with_columns(pl.lit(name).alias('window'))
parts=[]
for i,p in enumerate(sorted(Path('artifacts/detail_features').glob('*.parquet'))):
    table=p.stem; f=tf.get(table,0)
    for phase in ['development','evaluation']:
        d=hand_matrix(table,phase,known)
        if d is None: continue
        Xd=d.select(seqcols).to_numpy()
        hs=models[f].predict_proba(Xd,thread_count=6)[:,1] if phase=='development' else np.mean([m.predict_proba(Xd,thread_count=6)[:,1] for m in models],axis=0)
        z=d.select('pair_id','hand_id').with_columns(pl.Series('hs',hs)).join(hands.select('hand_id','idx'),on='hand_id')
        if phase=='development':
            for w,(lo,hi) in WINDOWS.items():
                if w=='evaluation': continue
                parts.append(agg(z.filter((C('idx')>=lo)&(C('idx')<hi)),w))
        else: parts.append(agg(z,'evaluation'))
    if i%40==0: print(i,table,round(time.time()-t0),flush=True)
res=pl.concat(parts); res.write_parquet(out/'hand_score_features.parquet'); print('done',res.shape,round(time.time()-t0),flush=True)
