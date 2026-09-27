\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from scipy.special import logit,expit
from catboost import CatBoostClassifier,Pool
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session4_set_evidence import design,UNAMES,FAMILIES,ap
from evidence_data import load
ROOT=Path('artifacts/evidence_session5');OLD=Path('artifacts/evidence_session4');P=Path('artifacts/policy');C=pl.col
BACKEND=os.environ.get('PAIRWISE_BACKEND','cat');PREFIX='pairwise' if BACKEND=='cat' else 'pairwise_'+BACKEND
CONTEXT_ONLY=os.environ.get('PAIRWISE_CONTEXT_ONLY')=='1'
if CONTEXT_ONLY:PREFIX+='_context'
d,_=load();extra=pl.read_parquet(list((P/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((P/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
d=d.join(extra,on=['pair_id','hand_id']).join(pl.read_parquet(OLD/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
cols=json.loads((P/'relationship_evidence/columns.json').read_text());rows=[];start=time.time()
def pairx(x,i,j):return np.concatenate([x[i]-x[j],(x[i]+x[j])/2],1)
for f in range(4):
    q=d.join(pl.read_parquet(OLD/f'nested_blend/nested_outer{f}.parquet'),on=['pair_id','hand_id']);weights=np.load(OLD/f'nested_blend/unary_weights_fold{f}.npy')
    for bi,family in enumerate(FAMILIES):
        base=CatBoostClassifier();base.load_model(str(OLD/f'base_{family}_fold{f}.cbm'))
        selected=[cols[i] for i in np.argsort(base.feature_importances_)[-48:]]
        if CONTEXT_ONLY:selected=[]
        bags=[]
        for _,g in q.filter(C('behavior_family')==family).group_by('pair_id'):
            b=design(g.with_columns((C('relative_time')*.6).alias('time')))
            z=pl.DataFrame({'hand_id':b['hand']}).join(g,on='hand_id',maintain_order='left',validate='1:1')
            assert z['hand_id'].to_list()==b['hand']
            b['x']=np.concatenate([b['U'],z.select(selected).to_numpy()],1).astype('float32') if selected else b['U']
            bags.append(b)
        xx=[];yy=[];ww=[];offsets=[]
        for b in bags:
            if b['fold']==f:continue
            i,j=np.where(b['y'][:,None]!=b['y'][None,:])
            if len(i)==0:continue
            xx.append(pairx(b['x'],i,j));yy.append(b['y'][i]);ww.append(np.full(len(i),1/len(i)))
            n=len(UNAMES);prior=b['U']@(weights[:n]+weights[n+bi*n:n+(bi+1)*n]);offsets.append(prior[i]-prior[j])
        if BACKEND in ['cat','residual']:
            m=CatBoostClassifier(iterations=350,depth=4,learning_rate=.025,l2_leaf_reg=20,thread_count=4,random_seed=5530+f,verbose=False,allow_writing_files=False)
            pool=Pool(np.concatenate(xx),np.concatenate(yy),weight=np.concatenate(ww),baseline=np.concatenate(offsets) if BACKEND=='residual' else None)
            m.fit(pool);m.save_model(str(ROOT/f'{PREFIX}_{family}_fold{f}.cbm'))
            if BACKEND=='residual':assert np.allclose(m.predict(pool,prediction_type='RawFormulaVal'),m.predict(np.concatenate(xx),prediction_type='RawFormulaVal')+np.concatenate(offsets),atol=1e-6)
        else:
            m=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.05,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=5530+f)
            w=np.concatenate(ww)
            with threadpool_limits(limits=4):m.fit(np.concatenate(xx),np.concatenate(yy),sample_weight=w/w.mean())
            joblib.dump(m,ROOT/f'{PREFIX}_{family}_fold{f}.joblib',compress=3)
        (ROOT/f'{PREFIX}_{family}_fold{f}_columns.json').write_text(json.dumps(selected))
        for b in bags:
            if b['fold']!=f:continue
            i,j=np.where(~np.eye(12,dtype=bool))
            n=len(UNAMES);prior=b['U']@(weights[:n]+weights[n+bi*n:n+(bi+1)*n])
            with threadpool_limits(limits=4):
                if BACKEND=='residual':p=expit(m.predict(pairx(b['x'],i,j),prediction_type='RawFormulaVal')+prior[i]-prior[j])
                else:p=m.predict_proba(pairx(b['x'],i,j))[:,1]
            matrix=np.zeros((12,12));matrix[i,j]=p;matrix=(matrix+1-matrix.T)/2
            comparator=logit(np.clip((matrix.sum(1)-.5)/11,1e-5,1-1e-5))
            n=len(UNAMES);prior=b['U']@(weights[:n]+weights[n+bi*n:n+(bi+1)*n])
            row={'pair_id':b['pid'],'fold':f,'family':family}
            for name,s in [('r27',prior),('comparator',comparator),('comparator_quarter',.75*prior+.25*comparator),('comparator_half',.5*prior+.5*comparator)]:
                row[name]=ap(b,np.lexsort((np.array(b['hand']),-s))[:5])
            rows.append(row)
    print('pairwise fold',f,'seconds',round(time.time()-start,1),flush=True)
r=pl.DataFrame(rows);r.write_csv(ROOT/f'{PREFIX}_comparison.csv');stats={n:{'map5':r[n].mean(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows())} for n in ['r27','comparator','comparator_quarter','comparator_half']}
(ROOT/f'{PREFIX}_comparison.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
