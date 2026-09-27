import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from evidence_data import load
root=Path('artifacts/policy'); C=pl.col
S='cache/'
lab=pl.read_csv('data/development_labels.csv')
                                                                                               
seqcols=json.loads((root/'sequence/evidence_columns.json').read_text())
h=pl.scan_parquet(str(root/'hand_features/*.parquet')).filter(C('phase')=='development').join(lab.lazy().select('pair_id'),on='pair_id').collect().sort('pair_id','time_index'); rc=[c for c in h.columns if c.endswith('_r')]
h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]); hz=[c for c in h.columns if c.endswith('_hz')]
h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]); add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
from sequence_features import augment
d=pl.read_parquet('artifacts/dev_detail.parquet').join(lab.select('pair_id','label','behavior_family'),on='pair_id').join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']).join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').with_columns(pl.lit(1).alias('evidence')),on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0),(C('time')/.6).alias('relative_time'))
d,_=augment(d); d=d.sort('pair_id','hand_id')
print('hands',d.height,'pairs',d['pair_id'].n_unique(),'evidence',int(d['evidence'].sum()),flush=True)
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
d=d.with_columns(pl.Series('fold',[tf.get(t,-1) for t in d['table_id']]))
X=d.select(seqcols).to_numpy(); y=d['evidence'].to_numpy(); lab_=d['label'].to_numpy(); fold=d['fold'].to_numpy()
                                                                                                                   
trmask=(y==1)|(lab_==0)
score=np.zeros(len(d))
for f in range(4):
    tr=trmask&(fold!=f); va=fold==f
    m=CatBoostClassifier(iterations=800,depth=6,learning_rate=.04,loss_function='Logloss',l2_leaf_reg=8,thread_count=6,random_seed=5+f,verbose=False,allow_writing_files=False,scale_pos_weight=float((lab_[tr]==0).sum()/max(1,(y[tr]==1).sum())))
    m.fit(X[tr],y[tr]); score[va]=m.predict_proba(X[va])[:,1]; print('fold',f,flush=True)
d=d.with_columns(pl.Series('hs',score))
d.select('pair_id','hand_id','label','evidence','time','hs').write_parquet(S+'hand_probe_scores.parquet')
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
d=d.join(hands.select('hand_id','idx'),on='hand_id'); evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
for w,(lo,hi) in {'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}.items():
    z=d.filter((C('idx')>=lo)&(C('idx')<hi))
    a=z.group_by('pair_id').agg(C('label').first(),pl.len().alias('n'),C('hs').max().alias('mx'),C('hs').top_k(3).mean().alias('t3'),C('hs').top_k(5).mean().alias('t5'),(C('hs')>.5).sum().alias('c5'),C('hs').sum().alias('sm'),(C('hs')/(1-C('hs')+1e-6)).log().clip(-6,6).top_k(5).sum().alias('lr5'))
    gw=evd.filter((C('idx')>=lo)&(C('idx')<hi)).group_by('pair_id').agg(pl.len().alias('n_ev_in')); a=a.join(gw,on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0)).filter((C('label')==0)|(C('n_ev_in')>=1))
    a=a.join(pl.read_parquet(S+f'allpairs_{w}.parquet').select('pair_id',C('risk').alias('v4')),on='pair_id')
    yy=(a['label']==1).to_numpy(); weak=yy&(a['n_ev_in']<=3).to_numpy(); v4=a['v4'].to_numpy(); wt=np.where(yy,1,50)
    print(f"{w}: pos={yy.sum()} | AP50 v4={ap(yy,v4,sample_weight=wt):.4f}",' '.join(f"{c}={ap(yy,a[c].to_numpy(),sample_weight=wt):.4f}" for c in ['mx','t3','t5','c5','lr5']),f"| weak-vs-neg AUC v4={auc(np.r_[np.ones(weak.sum()),np.zeros((~yy).sum())],np.r_[v4[weak],v4[~yy]]):.4f}",' '.join(f"{c}={auc(np.r_[np.ones(weak.sum()),np.zeros((~yy).sum())],np.r_[a[c].to_numpy()[weak],a[c].to_numpy()[~yy]]):.4f}" for c in ['t3','lr5']))
    bur=yy&(v4<0.5); t3=a['t3'].to_numpy(); negs=np.sort(t3[~yy]); above=len(negs)-np.searchsorted(negs,t3[bur],side='right')
    print(f"   v4-buried positives {bur.sum()}: negatives above by t3 median {np.median(above) if bur.sum() else 'NA'} of {len(negs)}; <=5 above: {np.mean(above<=5) if bur.sum() else 'NA'}")
