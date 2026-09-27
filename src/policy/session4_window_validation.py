\
\
\
\
\
import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');sys.path.insert(0,'src/policy')
import numpy as np,polars as pl,json,time
from pathlib import Path
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sequence_features import augment
from session4_evidence_model import load_models,score,correct,COLS,ROOT,FAMILIES
models=load_models();old={}
for b in FAMILIES:
    old[b]=[]
    for f in range(4):
        m=CatBoostClassifier();m.load_model(f'artifacts/policy/relationship_evidence/{b}_fold{f}.cbm');old[b].append(m)
ix=pl.read_parquet(ROOT/'hand_index.parquet');F=pl.read_parquet('artifacts/evidence_windows/window_features.parquet')
F=F.join(ix.select('pair_id','hand_id','time','fold','net_direction'),on=['pair_id','hand_id'])
rows=[];t=time.time()
with threadpool_limits(limits=4):
    for w,lo,hi in [('first_2000',0,2000),('last_2000',1000,3000)]:
        z=F.filter(pl.col('window')==w).with_columns(pl.lit('development').alias('phase'),((pl.col('time')*5000-lo)/(hi-lo)).alias('relative_time'))
        z,_=augment(z)
        for f in range(4):
            weights=np.load(ROOT/f'nested_blend/unary_weights_fold{f}.npy')
            for b in FAMILIES:
                q=z.filter((pl.col('fold')==f)&(pl.col('behavior_family')==b));x=q.select(COLS).to_numpy()
                if not len(q):continue
                q=q.with_columns(pl.Series('base_score',score(models,b,x,[f])),pl.Series('old_score',old[b][f].predict_proba(x,thread_count=4)[:,1]))
                for (pid,),g in q.group_by('pair_id'):
                    true=set(g.filter(pl.col('evidence')==1)['hand_id']);den=min(5,len(true))
                    if not den:continue
                    ranks={'production':g.sort(['old_score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5],
                           'blend':g.sort(['base_score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5],
                           'adjusted':correct(g,weights)}
                    row={'pair_id':pid,'table_id':g['table_id'][0],'fold':f,'window':w,'family':b}
                    for name,hands in ranks.items():
                        y=np.array([h in true for h in hands]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
                    rows.append(row)
        print('window',w,'seconds',round(time.time()-t,1),flush=True)
r=pl.DataFrame(rows);r.write_csv(ROOT/'window_validation.csv');print(r.group_by('window').agg(pl.len(),pl.col('production','blend','adjusted').mean()))
print(r.group_by('window','family').agg(pl.col('production','blend','adjusted').mean()))
