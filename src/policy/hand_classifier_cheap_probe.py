import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6')
import json,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
C=pl.col; S='cache/'
lab=pl.read_csv('data/development_labels.csv')
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at','big_blind','final_pot','players_at_showdown']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id','net_chips','folded','went_to_showdown','total_contribution','seat_no'])
def hand_table(table,pairs,phase):
    h=pl.read_parquet(S+f'hand_rows/{table}.parquet').filter(C('phase')==phase).join(pairs,on='pair_id').sort('pair_id','time_index')
    if h.is_empty(): return None
    rc=[c for c in h.columns if c.endswith('_r')]
    h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]); hz=[c for c in h.columns if c.endswith('_hz')]
    h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz],*[C(c).rolling_mean(21,min_samples=1,center=True).over('pair_id').alias(c+'_near21') for c in hz])
    s1=seats.rename({'player_id':'player_1'}).select('hand_id','player_1',C('net_chips').alias('net1'),C('folded').alias('f1'),C('went_to_showdown').alias('sd1'),C('total_contribution').alias('put1'),C('seat_no').alias('seat1'))
    s2=seats.rename({'player_id':'player_2'}).select('hand_id','player_2',C('net_chips').alias('net2'),C('folded').alias('f2'),C('went_to_showdown').alias('sd2'),C('total_contribution').alias('put2'),C('seat_no').alias('seat2'))
    h=h.join(s1,on=['hand_id','player_1']).join(s2,on=['hand_id','player_2']).join(hands.select('hand_id','big_blind','final_pot','players_at_showdown','idx'),on='hand_id')
    h=h.with_columns((C('final_pot')/C('big_blind')).alias('pot_bb'),(C('net1')/C('big_blind')).alias('net1_bb'),(C('net2')/C('big_blind')).alias('net2_bb'),((C('net1')-C('net2')).abs()/C('big_blind')).alias('netdiff_bb'),(C('put1')/C('big_blind')).alias('put1_bb'),(C('put2')/C('big_blind')).alias('put2_bb'),(~C('f1')&~C('f2')).cast(pl.Int8).alias('both_unf'),(C('sd1')&C('sd2')).cast(pl.Int8).alias('both_sd'),(C('f1').cast(pl.Int8)+C('f2').cast(pl.Int8)).alias('n_fold'),((C('seat1')-C('seat2'))%6).alias('seatd'))
    return h
feat=None
parts=[]
for table in sorted({t for t in pl.read_parquet(S+'allpairs_full.parquet')['table_id'].unique()}):
    h=hand_table(table,lab.select('pair_id','label'),'development')
    if h is None: continue
    parts.append(h)
d=pl.concat(parts).join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0))
feat=[c for c in d.columns if c.endswith(('_hz','_near5','_near21','_r','_v')) or c in ['surprise_max','pot_bb','net1_bb','net2_bb','netdiff_bb','put1_bb','put2_bb','both_unf','both_sd','n_fold','seatd','players_at_showdown']]
print('hands',d.height,'features',len(feat),'evidence',int(d['evidence'].sum()),flush=True)
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
tabs=pl.read_parquet(S+'allpairs_full.parquet').select('pair_id','table_id'); d=d.join(tabs,on='pair_id').with_columns(pl.Series('fold',[tf.get(t,-1) for t in d.join(tabs,on='pair_id')['table_id']]))
X=d.select(feat).to_numpy(); y=d['evidence'].to_numpy(); lab_=d['label'].to_numpy(); fold=d['fold'].to_numpy(); trmask=(y==1)|(lab_==0); score=np.zeros(len(d))
for f in range(4):
    tr=trmask&(fold!=f); va=fold==f
    m=CatBoostClassifier(iterations=800,depth=6,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=8,thread_count=6,random_seed=5+f,verbose=False,allow_writing_files=False,scale_pos_weight=float((lab_[tr]==0).sum()/max(1,(y[tr]==1).sum())))
    m.fit(X[tr],y[tr]); score[va]=m.predict_proba(X[va])[:,1]; print('fold',f,flush=True)
d=d.with_columns(pl.Series('hs',score)); d.select('pair_id','hand_id','label','evidence','idx','hs').write_parquet(S+'hand_cheap_scores.parquet')
evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
for w,(lo,hi) in {'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}.items():
    z=d.filter((C('idx')>=lo)&(C('idx')<hi))
    a=z.group_by('pair_id').agg(C('label').first(),C('hs').top_k(3).mean().alias('t3'),(C('hs')/(1-C('hs')+1e-6)).log().clip(-6,6).top_k(5).sum().alias('lr5'))
    gw=evd.filter((C('idx')>=lo)&(C('idx')<hi)).group_by('pair_id').agg(pl.len().alias('n_ev_in')); a=a.join(gw,on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0)).filter((C('label')==0)|(C('n_ev_in')>=1))
    a=a.join(pl.read_parquet(S+f'allpairs_{w}.parquet').select('pair_id',C('risk').alias('v4')),on='pair_id')
    yy=(a['label']==1).to_numpy(); weak=yy&(a['n_ev_in']<=3).to_numpy(); v4=a['v4'].to_numpy(); wt=np.where(yy,1,50)
    print(f"{w}: AP50 v4={ap(yy,v4,sample_weight=wt):.4f} t3={ap(yy,a['t3'].to_numpy(),sample_weight=wt):.4f} lr5={ap(yy,a['lr5'].to_numpy(),sample_weight=wt):.4f} | weak AUC v4={auc(np.r_[np.ones(weak.sum()),np.zeros((~yy).sum())],np.r_[v4[weak],v4[~yy]]):.4f} t3={auc(np.r_[np.ones(weak.sum()),np.zeros((~yy).sum())],np.r_[a['t3'].to_numpy()[weak],a['t3'].to_numpy()[~yy]]):.4f}")
    bur=yy&(v4<0.5); t3=a['t3'].to_numpy(); negs=np.sort(t3[~yy]); above=len(negs)-np.searchsorted(negs,t3[bur],side='right'); print(f"   v4-buried {bur.sum()}: negs above by t3 median {np.median(above) if bur.sum() else 'NA'}")
