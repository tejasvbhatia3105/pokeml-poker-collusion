\
\
\
\
import os,json,time,joblib
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from scipy.special import logit
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session4_set_evidence import design,UNAMES,FAMILIES,ap
ROOT=Path('artifacts/evidence_session5');OLD=Path('artifacts/evidence_session4');C=pl.col
d=pl.read_parquet('artifacts/policy/evidence_training.parquet').join(pl.read_parquet(OLD/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id']);rows=[];start=time.time()
with threadpool_limits(limits=4):
    for f in range(4):
        q=d.join(pl.read_parquet(OLD/f'nested_blend/nested_outer{f}.parquet'),on=['pair_id','hand_id']);bags=[design(g.with_columns((C('relative_time')*.6).alias('time'))) for _,g in q.group_by('pair_id')]
        for b in bags:b['x']=np.column_stack([b['U'],np.full(12,FAMILIES.index(b['family']))])
        tr=[b for b in bags if b['fold']!=f];X=np.concatenate([b['x'] for b in tr]);y=np.concatenate([b['y'] for b in tr]);weights=np.load(OLD/f'nested_blend/unary_weights_fold{f}.npy')
        models={'cat':CatBoostClassifier(iterations=300,depth=3,learning_rate=.025,l2_leaf_reg=20,thread_count=4,random_seed=5790+f,verbose=False,allow_writing_files=False),
                'hist':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=5790+f)}
        for name,m in models.items():
            m.fit(X,y)
            if name=='cat':m.save_model(str(ROOT/f'nonlinear_{name}_fold{f}.cbm'))
            else:joblib.dump(m,ROOT/f'nonlinear_{name}_fold{f}.joblib',compress=3)
        for b in bags:
            if b['fold']!=f:continue
            n=len(UNAMES);bi=FAMILIES.index(b['family']);prior=b['U']@(weights[:n]+weights[n+bi*n:n+(bi+1)*n]);scores={name:logit(np.clip(m.predict_proba(b['x'])[:,1],1e-5,1-1e-5)) for name,m in models.items()}
            row={'pair_id':b['pid'],'fold':f,'family':b['family'],'r27':ap(b,np.lexsort((np.array(b['hand']),-prior))[:5])}
            for name,s in scores.items():
                for alpha in [.25,1.]:row[f'{name}_{alpha}']=ap(b,np.lexsort((np.array(b['hand']),-((1-alpha)*prior+alpha*s)))[:5])
            rows.append(row)
        print('nonlinear fold',f,'seconds',round(time.time()-start,1),flush=True)
r=pl.DataFrame(rows);r.write_csv(ROOT/'nonlinear_context_comparison.csv');stats={n:{'map5':r[n].mean(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows())} for n in r.columns if n not in ['pair_id','fold','family']}
(ROOT/'nonlinear_context_comparison.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
