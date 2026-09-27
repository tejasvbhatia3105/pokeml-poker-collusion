\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
from sequence_features import augment
root=Path('artifacts/policy'); C=pl.col; NAMES=['directed_transfer','soft_play','coordinated_isolation']
oldcols=json.loads(Path('artifacts/rank_columns.json').read_text()); newcols=json.loads((root/'sequence/evidence_columns.json').read_text())
tf=json.loads((root/'table_folds.json').read_text())
models={n:{k:[] for k in ['old','sequence']} for n in NAMES}
for n in NAMES:
    for f in range(4):
        m=CatBoostClassifier(); m.load_model(f'artifacts/rank_{n}_fold{f}.cbm'); models[n]['old'].append(m)
        m=CatBoostClassifier(); m.load_model(str(root/f'sequence/evidence_{n}_fold{f}.cbm')); models[n]['sequence'].append(m)
lab=pl.read_csv('data/development_labels.csv')
dd=pl.read_parquet('artifacts/dev_detail.parquet').with_columns((C('time')/.6).alias('relative_time'))
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
parts=[]
for table,q in dd.group_by('table_id'):
    table=table[0]
    h=pl.read_parquet(root/'hand_features'/f'{table}.parquet').filter(C('phase')=='development').join(q.select('pair_id').unique(),on='pair_id').sort('pair_id','time_index'); rc=[c for c in h.columns if c.endswith('_r')]
    h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]); hz=[c for c in h.columns if c.endswith('_hz')]
    h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]); add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
    d=q.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']); d,_=augment(d)
    f=tf[table]; out=d.select('pair_id','hand_id')
    for n in NAMES:
        op=models[n]['old'][f].predict_proba(d.select(oldcols).to_numpy(),thread_count=6)[:,1]; sp=models[n]['sequence'][f].predict_proba(d.select(newcols).to_numpy(),thread_count=6)[:,1]
        out=out.with_columns(pl.Series('ev_'+n,.25*op+.75*sp))
    parts.append(out)
s=pl.concat(parts).join(hands.select('hand_id','idx'),on='hand_id'); s.write_parquet(root/'evidence_hand_scores_labelled.parquet'); print('scored hands',s.height,flush=True)
def agg(s,lo,hi):
    z=s.filter((C('idx')>=lo)&(C('idx')<hi)); ex=[pl.len().alias('n_hands')]
    for n in NAMES:
        c=C('ev_'+n); ex+=[c.max().alias(f'{n}_max'),c.top_k(3).mean().alias(f'{n}_top3'),c.top_k(5).mean().alias(f'{n}_top5'),(c>.3).sum().alias(f'{n}_n30'),(c>.6).sum().alias(f'{n}_n60'),c.mean().alias(f'{n}_mean')]
    return z.group_by('pair_id').agg(ex)
S='cache/'
evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
for w,(lo,hi) in {'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}.items():
    a=agg(s,lo,hi).join(lab.select('pair_id','label'),on='pair_id').join(pl.read_parquet(S+f'allpairs_{w}.parquet').select('pair_id',C('risk').alias('v4')),on='pair_id')
    gw=evd.filter((C('idx')>=lo)&(C('idx')<hi)).group_by('pair_id').agg(pl.len().alias('n_ev_in')); a=a.join(gw,on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0))
    a=a.filter((C('label')==0)|(C('n_ev_in')>=1)); y=(a['label']==1).to_numpy(); weak=y&(a['n_ev_in']<=3).to_numpy(); v4=a['v4'].to_numpy()
    best=np.max(np.stack([a[f'{n}_top3'].to_numpy() for n in NAMES]),0); bmax=np.max(np.stack([a[f'{n}_max'].to_numpy() for n in NAMES]),0)
    print(f"{w}: n={a.height} pos={y.sum()} | AP50 v4={ap(y,v4,sample_weight=np.where(y,1,50)):.4f} ev_top3={ap(y,best,sample_weight=np.where(y,1,50)):.4f} ev_max={ap(y,bmax,sample_weight=np.where(y,1,50)):.4f} | weak vs neg AUC v4={auc(np.r_[np.ones(weak.sum()),np.zeros((~y).sum())],np.r_[v4[weak],v4[~y]]):.4f} ev_top3={auc(np.r_[np.ones(weak.sum()),np.zeros((~y).sum())],np.r_[best[weak],best[~y]]):.4f}")
                                                                                      
    bur=y&(v4<0.5); negs=np.sort(best[~y]); above=len(negs)-np.searchsorted(negs,best[bur],side='right')
    print(f"   v4-buried positives {bur.sum()}: negatives above them by ev_top3 -> median {np.median(above) if bur.sum() else 'NA'}, of {len(negs)}; fraction with <=5 negatives above: {np.mean(above<=5) if bur.sum() else 'NA'}")
