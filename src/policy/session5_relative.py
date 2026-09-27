\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from scipy.stats import rankdata
from catboost import CatBoostClassifier
from evidence_data import load
ROOT=Path('artifacts/evidence_session5');OLD=Path('artifacts/evidence_session4');P=Path('artifacts/policy');C=pl.col
def transform(x,groups):
    out=np.empty((len(x),2*x.shape[1]),np.float32)
    for indices in groups:
        a=x[indices];lo,med,hi=np.quantile(a,[.25,.5,.75],axis=0)
        out[indices]=np.concatenate([rankdata(a,axis=0,method='average')/len(a),np.clip((a-med)/(hi-lo+1e-3),-10,10)],1)
    return out
def main():
    d,_=load();extra=pl.read_parquet(list((P/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((P/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(extra,on=['pair_id','hand_id']).join(pl.read_parquet(OLD/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id']).with_row_index('row')
    cols=json.loads((P/'relationship_evidence/columns.json').read_text());X=d.select(cols).to_numpy();y=d['evidence'].to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy()
    groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id')];pred=np.full(len(d),np.nan);start=time.time()
    for f in range(4):
        for family in ['directed_transfer','soft_play','coordinated_isolation']:
            m=CatBoostClassifier();m.load_model(str(OLD/f'base_{family}_fold{f}.cbm'));selected=np.argsort(m.feature_importances_)[-64:];new=transform(X[:,selected],groups);xx=np.concatenate([X,new],1)
            tr=(fv!=f)&(fam==family);va=(fv==f)&(fam==family)
            m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=1710+11*f,verbose=False,allow_writing_files=False)
            m.fit(xx[tr],y[tr]);pred[va]=m.predict_proba(xx[va],thread_count=4)[:,1];m.save_model(str(ROOT/f'relative_{family}_fold{f}.cbm'))
            (ROOT/f'relative_{family}_fold{f}_columns.json').write_text(json.dumps([cols[i] for i in selected]))
        print('relative fold',f,'seconds',round(time.time()-start,1),flush=True)
    d.select('pair_id','hand_id').with_columns(pl.Series('score',pred)).write_parquet(ROOT/'relative_oof.parquet')
if __name__=='__main__':main()
