\
\
\
\
\
import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');sys.path.insert(0,'src/policy')
import json,time
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from evidence_data import load

OUT=Path('artifacts/evidence_session4');root=Path('artifacts/policy');C=pl.col
def build():
    config=json.loads(Path('artifacts/seq_v6/config.json').read_text())
    extra=Path(config['tokens3']);v1=extra.parent/'seq_tokens'
    index=pl.read_parquet(OUT/'hand_index.parquet');parts=[];t=time.time()
    cols=json.loads((extra/'columns3.json').read_text())
    for (table,),q in index.group_by('table_id'):
        with np.load(v1/f'{table}.npz') as z:
            meta=pl.DataFrame({'row':np.arange(len(z['hand_id'])),'hand_id':z['hand_id'],
                              'pair_id':np.repeat(z['pair_id'],z['n'])})
            m=q.select('pair_id','hand_id').join(meta,on=['pair_id','hand_id'],validate='1:1')
        assert len(m)==len(q)
        x=np.asarray(np.load(extra/f'{table}.npy',mmap_mode='r')[m['row'].to_numpy()],dtype=np.float32)
                                                                                
        a=x[:,:17];b=x[:,17:34];r1=x[:,34:35];r2=x[:,35:36]
        a=a*r1;b=b*r2
        f=np.concatenate([np.minimum(a,b),np.maximum(a,b),a*b,x[:,36:38].sum(1,keepdims=True),
                          np.minimum(r1,r2),np.maximum(r1,r2)],1)
        names=[f'attribution_{kind}_{n}' for kind in ['min','max','product'] for n in cols[:17]]+['attribution_dealt','attribution_risk_min','attribution_risk_max']
        parts.append(m.select('pair_id','hand_id').hstack(pl.DataFrame(f,schema=names)))
    pl.concat(parts).write_parquet(OUT/'attribution_features.parquet')
    print('attribution cache built',round(time.time()-t,1),flush=True)
def main():
    if not (OUT/'attribution_features.parquet').exists():build()
    d,_=load();extra=pl.read_parquet(list((root/'outcome_roles').glob('T*.parquet'))).join(
        pl.read_parquet(list((root/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(extra,on=['pair_id','hand_id']).join(pl.read_parquet(OUT/'attribution_features.parquet'),on=['pair_id','hand_id'])
    d=d.join(pl.read_parquet(OUT/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
    basecols=json.loads((root/'relationship_evidence/columns.json').read_text())
    cols=basecols+[c for c in d.columns if c.startswith('attribution_')]
    y=d['evidence'].to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();parts=[];t=time.time()
    for f in range(4):
        pr=np.full(len(d),np.nan)
        for b in ['directed_transfer','soft_play','coordinated_isolation']:
            tr=(fv!=f)&(fam==b);va=(fv==f)&(fam==b)
            m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,loss_function='Logloss',l2_leaf_reg=8,
                thread_count=4,random_seed=1710+11*f,verbose=False,allow_writing_files=False)
            X=d.select(cols).to_numpy();m.fit(X[tr],y[tr]);pr[va]=m.predict_proba(X[va])[:,1]
            m.save_model(str(OUT/f'attribution_{b}_fold{f}.cbm'))
        va=fv==f
        parts.append(d.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence','fold').with_columns(pl.Series('score',pr[va])))
        print('attribution fold',f,round(time.time()-t,1),flush=True)
    pl.concat(parts).write_parquet(OUT/'attribution_oof.parquet')
    (OUT/'attribution_columns.json').write_text(json.dumps(cols))
if __name__=='__main__':main()
