import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,numpy as np,polars as pl,pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier,CatBoostRanker,Pool
from evidence_data import load
root=Path('artifacts/policy'); C=pl.col; K=int(sys.argv[1]) if len(sys.argv)>1 else 20
d,oldcols=load()
extra=pl.read_parquet(list((root/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((root/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'])
d=d.join(extra,on=['pair_id','hand_id'],how='left')
relcols=json.loads((root/'relationship_evidence/columns.json').read_text())
oof=pl.read_parquet(root/'relationship_evidence/oof.parquet').select('pair_id','hand_id',C('score').alias('s1'),C('v4').alias('s0'))
d=d.join(oof,on=['pair_id','hand_id'])
                               
d=d.sort(['pair_id','s1'],descending=[False,True]).with_columns(
    pl.int_range(pl.len()).over('pair_id').alias('rk'),(C('s1')/C('s1').max().over('pair_id')).alias('s1_rel'),(C('s1')-C('s1').shift(1).over('pair_id')).fill_null(0).alias('s1_gap'),
    C('s1').max().over('pair_id').alias('s1_max'),C('s1').mean().over('pair_id').alias('s1_mean'),(C('s0')/C('s0').max().over('pair_id')).alias('s0_rel'),
    C('hand_id').count().over('pair_id').alias('n_shared'),(C('s1')>0.5*C('s1').max().over('pair_id')).sum().over('pair_id').alias('n_strong'))
d=d.sort(['pair_id','time']).with_columns(pl.int_range(pl.len()).over('pair_id').alias('tk')).with_columns((C('tk')/C('n_shared')).alias('trel'))
                                                               
top=d.filter(C('rk')<K).sort(['pair_id','time']).with_columns(pl.int_range(pl.len()).over('pair_id').alias('ck'),C('hand_id').count().over('pair_id').alias('nc'))
top=top.with_columns((C('time')-C('time').shift(1).over('pair_id')).fill_null(1.0).alias('dt_prev'),(C('time').shift(-1).over('pair_id')-C('time')).fill_null(1.0).alias('dt_next'))
top=top.with_columns(pl.min_horizontal('dt_prev','dt_next').alias('dt_min'),((C('time')-C('time').min().over('pair_id'))/(C('time').max().over('pair_id')-C('time').min().over('pair_id')+1e-9)).alias('t_in_cands'))
newc=['rk','s1','s0','s1_rel','s1_gap','s1_max','s1_mean','s0_rel','n_shared','n_strong','trel','dt_prev','dt_next','dt_min','t_in_cands','ck','nc']
cols=relcols+newc
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
top=top.with_columns(pl.Series('fold',[tf.get(t,-1) for t in top['table_id']]))
truth_counts=dict(pl.read_csv('data/development_evidence.csv').group_by('pair_id').len().iter_rows())
def map5(df,col):
    out={}
    for pid,g in df.sort(['pair_id',col,'hand_id'],descending=[False,True,False]).group_by('pair_id',maintain_order=True):
        key=pid[0];r=g['evidence'].to_numpy()[:5]
        out[key]=float((np.cumsum(r)/np.arange(1,len(r)+1)*r).sum()/min(5,truth_counts[key]))
    return np.mean([out.get(pid,0.0) for pid in truth_counts])
                                                                                                                                                                    
full_map=map5(d,'s1'); print('first-stage MAP@5 (all hands):',round(full_map,4),' within top-%d only:'%K,round(map5(top,'s1'),4),flush=True)
res={}
for behavior_split in [True,False]:
    pred=np.zeros(len(top)); predr=np.zeros(len(top))
    for f in range(4):
        va=(top['fold']==f).to_numpy(); tr=~va
        groups=[('all',pl.lit(True))] if not behavior_split else [(b,C('behavior_family')==b) for b in ['directed_transfer','soft_play','coordinated_isolation']]
        for name,mask in groups:
            z=top.with_columns(mask.alias('m')); m=z['m'].to_numpy(); X=z.select(cols).to_numpy(); y=z['evidence'].to_numpy()
            clf=CatBoostClassifier(iterations=600,depth=5,learning_rate=.03,loss_function='Logloss',l2_leaf_reg=8,thread_count=6,random_seed=77+f,verbose=False,allow_writing_files=False)
            clf.fit(X[tr&m],y[tr&m]); pred[va&m]=clf.predict_proba(X[va&m])[:,1]
                                                
            zz=z.filter(pl.Series(tr&m)).sort('pair_id'); gid=zz['pair_id'].to_numpy()
            rk=CatBoostRanker(iterations=600,depth=5,learning_rate=.03,loss_function='YetiRank',thread_count=6,random_seed=77+f,verbose=False,allow_writing_files=False)
            rk.fit(Pool(zz.select(cols).to_numpy(),zz['evidence'].to_numpy(),group_id=gid)); predr[va&m]=rk.predict(X[va&m])
        print('fold',f,'done',flush=True)
    t2=top.with_columns(pl.Series('p2',pred),pl.Series('pr',predr),(0.5*pl.Series('p2',pred)+0.5*C('s1')).alias('pblend'))
    for c in ['p2','pr','pblend']:
        res[(behavior_split,c)]=map5(t2,c); print('behavior_split' if behavior_split else 'pooled',c,'MAP@5 (top-%d set):'%K,round(res[(behavior_split,c)],4),flush=True)
    if not behavior_split: t2.select('pair_id','hand_id','evidence','s1','p2','pr','behavior_family').write_parquet(root/'evidence_rerank_oof.parquet')
