\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,time,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
exec(open('src/policy/hand_classifier_cheap_probe.py').read().split("feat=None")[0])
out=Path(sys.argv[1]); out.mkdir(exist_ok=True,parents=True); ROWS=sys.argv[2] if len(sys.argv)>2 else S+'hand_rows'; t0=time.time()
def hand_table2(table,pairs,phase):
    global S
    h=pl.read_parquet(f'{ROWS}/{table}.parquet').filter(C('phase')==phase).join(pairs,on='pair_id').sort('pair_id','time_index')
    if h.is_empty(): return None
    rc=[c for c in h.columns if c.endswith('_r')]
    h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]); hz=[c for c in h.columns if c.endswith('_hz')]
    h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz],*[C(c).rolling_mean(21,min_samples=1,center=True).over('pair_id').alias(c+'_near21') for c in hz])
    s1=seats.rename({'player_id':'player_1'}).select('hand_id','player_1',C('net_chips').alias('net1'),C('folded').alias('f1'),C('went_to_showdown').alias('sd1'),C('total_contribution').alias('put1'),C('seat_no').alias('seat1'))
    s2=seats.rename({'player_id':'player_2'}).select('hand_id','player_2',C('net_chips').alias('net2'),C('folded').alias('f2'),C('went_to_showdown').alias('sd2'),C('total_contribution').alias('put2'),C('seat_no').alias('seat2'))
    h=h.join(s1,on=['hand_id','player_1']).join(s2,on=['hand_id','player_2']).join(hands.select('hand_id','big_blind','final_pot','players_at_showdown','idx'),on='hand_id')
    return h.with_columns((C('final_pot')/C('big_blind')).alias('pot_bb'),(C('net1')/C('big_blind')).alias('net1_bb'),(C('net2')/C('big_blind')).alias('net2_bb'),((C('net1')-C('net2')).abs()/C('big_blind')).alias('netdiff_bb'),(C('put1')/C('big_blind')).alias('put1_bb'),(C('put2')/C('big_blind')).alias('put2_bb'),(~C('f1')&~C('f2')).cast(pl.Int8).alias('both_unf'),(C('sd1')&C('sd2')).cast(pl.Int8).alias('both_sd'),(C('f1').cast(pl.Int8)+C('f2').cast(pl.Int8)).alias('n_fold'),((C('seat1')-C('seat2'))%6).alias('seatd'))
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
tables=sorted(p.stem for p in Path(ROWS).glob('*.parquet'))
allp=pl.read_parquet(S+'allpairs_full.parquet').select('pair_id','player_1','player_2','label','table_id')
pos=lab.filter(C('label')==1); partner={}
for a,b in zip(pos['player_1'],pos['player_2']): partner.setdefault(a,set()).add(b); partner.setdefault(b,set()).add(a)
hn=allp.filter(pl.Series([(l==-1) and ((a in partner)!=(b in partner)) for a,b,l in zip(allp['player_1'],allp['player_2'],allp['label'])])).sample(3000,seed=3)
trpairs=pl.concat([lab.select('pair_id','label'),hn.select('pair_id').with_columns(pl.lit(2,dtype=pl.Int64).alias('label'))]).with_columns(C('label').cast(pl.Int64))                            
parts=[]
for table in tables:
    if table not in tf: continue
    h=hand_table2(table,trpairs,'development')
    if h is not None: parts.append(h.with_columns(pl.lit(table).alias('table_id')))
d=pl.concat(parts).join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0))
feat=[c for c in d.columns if c.endswith(('_hz','_near5','_near21','_r','_v')) or c in ['surprise_max','pot_bb','net1_bb','net2_bb','netdiff_bb','put1_bb','put2_bb','both_unf','both_sd','n_fold','seatd','players_at_showdown'] or c.startswith('surp_')]
json.dump(feat,open(out/'features.json','w')); print('train hands',d.height,'features',len(feat),'evidence',int(d['evidence'].sum()),'hardneg hands',int((d['label']==2).sum()),round(time.time()-t0),flush=True)
X=d.select(feat).to_numpy(); y=d['evidence'].to_numpy(); lab_=d['label'].to_numpy(); fold=np.array([tf[t] for t in d['table_id']]); trmask=(y==1)|(lab_==0)|(lab_==2); wt=np.where(lab_==2,0.5,1.0); models=[]
for f in range(4):
    tr=trmask&(fold!=f)
    m=CatBoostClassifier(iterations=800,depth=6,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=8,thread_count=6,random_seed=5+f,verbose=False,allow_writing_files=False,scale_pos_weight=float(wt[tr&(y==0)].sum()/max(1,(y[tr]==1).sum())))
    m.fit(X[tr],y[tr],sample_weight=wt[tr]); m.save_model(str(out/f'hand_fold{f}.cbm')); models.append(m); print('fold',f,round(time.time()-t0),flush=True)
WINDOWS={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}
def agg(z,name):
    c=C('hs'); return z.group_by('pair_id').agg(pl.len().alias('hs_n'),c.max().alias('hs_max'),c.top_k(3).mean().alias('hs_top3'),c.top_k(5).mean().alias('hs_top5'),(c>.5).sum().alias('hs_c50'),(c>.8).sum().alias('hs_c80'),c.mean().alias('hs_mean'),(c/(1-c+1e-6)).log().clip(-6,6).top_k(5).sum().alias('hs_lr5'),(c/(1-c+1e-6)).log().clip(-6,6).sum().alias('hs_lrsum')).with_columns(pl.lit(name).alias('window'))
allpairs=pl.concat([pl.read_parquet(f'{ROWS}/{t}.parquet').select('pair_id').unique() for t in tables]).unique(); res=[]
for i,table in enumerate(tables):
    f=tf.get(table,0)
    for phase in ['development','evaluation']:
        h=hand_table2(table,allpairs,phase)
        if h is None: continue
        Xh=h.select(feat).to_numpy(); hs=models[f].predict_proba(Xh,thread_count=6)[:,1] if phase=='development' else np.mean([m.predict_proba(Xh,thread_count=6)[:,1] for m in models],axis=0)
        z=h.select('pair_id','idx').with_columns(pl.Series('hs',hs))
        if phase=='development':
            for w,(lo,hi) in WINDOWS.items(): res.append(agg(z.filter((C('idx')>=lo)&(C('idx')<hi)),w))
        else: res.append(agg(z,'evaluation'))
    if i%40==0: print(i,table,round(time.time()-t0),flush=True)
r=pl.concat(res); r.write_parquet(out/'hand_score_features.parquet'); print('done',r.shape,round(time.time()-t0),flush=True)
