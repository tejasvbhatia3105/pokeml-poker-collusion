import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
import json,time,numpy as np,polars as pl,pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from evidence_data import load
root=Path('artifacts/policy'); C=pl.col; out=Path(sys.argv[1]); out.mkdir(exist_ok=True,parents=True); seeds=[int(x) for x in sys.argv[2].split(',')]; use_hs=len(sys.argv)>3 and sys.argv[3]=='hs'
S='cache/'
d,oldcols=load(); extra=pl.read_parquet(list((root/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((root/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
d=d.join(extra,on=['pair_id','hand_id'],how='left'); cols=json.loads((root/'relationship_evidence/columns.json').read_text())
if use_hs:
    hs=pl.read_parquet(S+'hand_cheap_scores.parquet').select('pair_id','hand_id','hs'); d=d.join(hs,on=['pair_id','hand_id'],how='left').with_columns(C('hs').fill_null(0)); d=d.with_columns((C('hs')/C('hs').max().over('pair_id')).alias('hs_rel'),(C('hs').rank(descending=True).over('pair_id')).alias('hs_rank')); cols=cols+['hs','hs_rel','hs_rank']
(out/'columns.json').write_text(json.dumps(cols)); folds=json.loads(Path('artifacts/folds.json').read_text()); parts=[]; t=time.time()
for f in folds:
    for behavior in ['directed_transfer','soft_play','coordinated_isolation']:
        q=d.filter(C('behavior_family')==behavior); X=q.select(cols).to_numpy(); y=q['evidence'].to_numpy(); va=q['table_id'].is_in(f['valid_tables']).to_numpy()
        pr=np.zeros(va.sum())
        for s in seeds:
            m=CatBoostClassifier(iterations=850,depth=5,learning_rate=.035,loss_function='Logloss',eval_metric='PRAUC',l2_leaf_reg=8,thread_count=6,random_seed=s+f['fold'],verbose=False,allow_writing_files=False)
            m.fit(X[~va],y[~va],eval_set=(X[va],y[va]),early_stopping_rounds=120); m.save_model(str(out/f'{behavior}_seed{s}_fold{f["fold"]}.cbm')); pr+=m.predict_proba(X[va])[:,1]/len(seeds)
        parts.append(q.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence').with_columns(pl.Series('score',pr)))
    print('fold',f['fold'],round(time.time()-t),flush=True)
o=pl.concat(parts).join(pl.read_parquet(root/'relationship_evidence/oof.parquet').select('pair_id','hand_id',C('score').alias('rel1')),on=['pair_id','hand_id']); o.write_parquet(out/'oof.parquet')
def map5(df,col):
    r=[]
    for pid,g in df.sort(['pair_id',col,'hand_id'],descending=[False,True,False]).group_by('pair_id',maintain_order=True):
        e=g['evidence'].to_numpy()[:5]; r.append(float((np.cumsum(e)/np.arange(1,len(e)+1)*e).sum()/min(5,g['evidence'].sum())))
    return np.mean(r)
print('MAP@5 single-seed relationship:',round(map5(o,'rel1'),4),' new:',round(map5(o,'score'),4),' mean(new,rel1):',round(map5(o.with_columns(((C('score')+C('rel1'))/2).alias('b')),'b'),4))
for b in ['directed_transfer','soft_play','coordinated_isolation']: print(' ',b,round(map5(o.filter(C('behavior_family')==b),'score'),4))
