import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,time,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
exec(open('src/policy/hand_classifier_cheap_probe.py').read().split("feat=None")[0])                                   
out=Path(sys.argv[1]); out.mkdir(exist_ok=True,parents=True); t0=time.time()
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
tables=sorted(p.stem for p in Path(S+'hand_rows').glob('*.parquet'))
                                         
parts=[]
for table in tables:
    h=hand_table(table,lab.select('pair_id','label'),'development')
    if h is not None: parts.append(h.with_columns(pl.lit(table).alias('table_id')))
d=pl.concat(parts).join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0))
feat=[c for c in d.columns if c.endswith(('_hz','_near5','_near21','_r','_v')) or c in ['surprise_max','pot_bb','net1_bb','net2_bb','netdiff_bb','put1_bb','put2_bb','both_unf','both_sd','n_fold','seatd','players_at_showdown']]
json.dump(feat,open(out/'features.json','w'))
X=d.select(feat).to_numpy(); y=d['evidence'].to_numpy(); lab_=d['label'].to_numpy(); fold=np.array([tf.get(t,-1) for t in d['table_id']]); trmask=(y==1)|(lab_==0); models=[]
for f in range(4):
    tr=trmask&(fold!=f)
    m=CatBoostClassifier(iterations=800,depth=6,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=8,thread_count=6,random_seed=5+f,verbose=False,allow_writing_files=False,scale_pos_weight=float((lab_[tr]==0).sum()/max(1,(y[tr]==1).sum())))
    m.fit(X[tr],y[tr]); m.save_model(str(out/f'hand_fold{f}.cbm')); models.append(m); print('fold',f,round(time.time()-t0),flush=True)
WINDOWS={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000),'w500_2500':(500,2500),'w0_1500':(0,1500),'w750_2250':(750,2250),'w1500_3000':(1500,3000)}
def agg(z,name):
    c=C('hs'); return z.group_by('pair_id').agg(pl.len().alias('hs_n'),c.max().alias('hs_max'),c.top_k(3).mean().alias('hs_top3'),c.top_k(5).mean().alias('hs_top5'),(c>.5).sum().alias('hs_c50'),(c>.8).sum().alias('hs_c80'),c.mean().alias('hs_mean'),(c/(1-c+1e-6)).log().clip(-6,6).top_k(5).sum().alias('hs_lr5'),(c/(1-c+1e-6)).log().clip(-6,6).sum().alias('hs_lrsum')).with_columns(pl.lit(name).alias('window'))
allpairs=pl.concat([pl.read_parquet(S+f'hand_rows/{t}.parquet').select('pair_id').unique() for t in tables]).unique()
res=[]
for i,table in enumerate(tables):
    f=tf.get(table,0)
    for phase in ['development','evaluation']:
        h=hand_table(table,allpairs,phase)
        if h is None: continue
        Xh=h.select(feat).to_numpy()
        hs=models[f].predict_proba(Xh,thread_count=6)[:,1] if phase=='development' else np.mean([m.predict_proba(Xh,thread_count=6)[:,1] for m in models],axis=0)
        z=h.select('pair_id','idx').with_columns(pl.Series('hs',hs))
        if phase=='development':
            for w,(lo,hi) in WINDOWS.items(): res.append(agg(z.filter((C('idx')>=lo)&(C('idx')<hi)),w))
        else: res.append(agg(z,'evaluation'))
    if i%40==0: print(i,table,round(time.time()-t0),flush=True)
r=pl.concat(res); r.write_parquet(out/'hand_score_features.parquet'); print('done',r.shape,round(time.time()-t0),flush=True)
