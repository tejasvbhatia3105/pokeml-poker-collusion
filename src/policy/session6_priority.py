\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from evidence_data import load
ROOT=Path('artifacts/evidence_session6');OLD=Path('artifacts/evidence_session4');P=Path('artifacts/policy');C=pl.col
def training_targets(y,sub,tp,tr,t,groups,rank=None):
    \
    assign=np.where(tr&(sub>0),sub,np.where(tp>=.5,1,2))
    if rank is not None:
        for ix in groups:
            if not tr[ix[0]] or np.any(sub[ix]>0):continue
            pos=ix[y[ix]==1];pos=pos[np.argsort(rank[pos],kind='stable')]
            p=np.clip(tp[pos],1e-6,1-1e-6)
            k=int(np.argmax(np.r_[0,np.cumsum(np.log(p)-np.log1p(-p))]))
            assign[pos[:k]]=1;assign[pos[k:]]=2
    target1=tr&(y==1)&(assign==1);target2=tr&(y==1)&(assign==2)
    eligible1=tr.copy();eligible2=tr.copy()
    for ix in groups:
        if not tr[ix[0]]:continue
        ntrue=y[ix].sum();n1=target1[ix].sum();n2=target2[ix].sum()
        if ntrue>=5:
            if n1>=5:eligible1[ix[t[ix]>t[ix[target1[ix]]].max()]]=False
            if n2==0:eligible2[ix]=False
            else:eligible2[ix[t[ix]>t[ix[target2[ix]]].max()]]=False
    return target1,target2,eligible1,eligible2
def inclusion(primary,secondary):
                                                                                  
                                                                          
    scale=np.maximum(1,primary+secondary);a=primary/scale;b=secondary/scale;n=len(a)
    def scan(prob):
        z=np.zeros((len(prob)+1,5));z[0,0]=1
        for i,p in enumerate(prob):z[i+1]=np.r_[z[i,0]*(1-p),z[i,1:]*(1-p)+z[i,:-1]*p]
        return z
    pa=scan(a);pany=scan(a+b);sa=scan(a[::-1])[::-1];out=np.empty(n)
    for i in range(n):out[i]=a[i]*pa[i].sum()+b[i]*sum(pany[i,k]*sa[i+1,:5-k].sum() for k in range(5))
    return out
def main():
    ordered=os.environ.get('PRIORITY_ORDERED','0')=='1';prefix='priority_ordered' if ordered else 'priority'
    d,_=load();extra=pl.read_parquet(list((P/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((P/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(extra,on=['pair_id','hand_id']).join(pl.read_parquet(OLD/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
    d=d.join(pl.read_parquet(ROOT/'order_subtype_labels.parquet'),on=['pair_id','hand_id'],how='left').with_columns(C('subtype').fill_null(0)).sort('pair_id','time','hand_id').with_row_index('row')
    d=d.join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id','evidence_rank'),on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left')
    rank=d['evidence_rank'].fill_null(0).to_numpy() if ordered else None
    cols=json.loads((P/'relationship_evidence/columns.json').read_text());typecols=[c for c in cols if not any(k in c for k in ['relationship_','preceding_','near5','time'])]
    X=d.select(cols).to_numpy();XT=d.select(typecols).to_numpy();y=d['evidence'].to_numpy();sub=d['subtype'].to_numpy();fv=d['fold'].to_numpy();family=d['behavior_family'].to_numpy();t=d['time'].to_numpy();groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)]
    preds=np.zeros((len(d),3));metrics=[];start=time.time()
    for f in range(4):
        for name in ['directed_transfer','soft_play','coordinated_isolation']:
            tr=(fv!=f)&(family==name);va=(fv==f)&(family==name);known=tr&(sub>0)
            typ=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,thread_count=4,random_seed=6210+f,verbose=False,allow_writing_files=False)
            typ.fit(XT[known],(sub[known]==1).astype(int));tp=typ.predict_proba(XT,thread_count=4)[:,1];typ.save_model(str(ROOT/f'{prefix}_type_{name}_fold{f}.cbm'))
            kv=va&(sub>0)
            if len(np.unique(sub[kv]))==2:metrics.append({'fold':f,'family':name,'known_validation_hands':int(kv.sum()),'subtype_auc':roc_auc_score(sub[kv]==1,tp[kv])})
                                                                                
            target1,target2,eligible1,eligible2=training_targets(y,sub,tp,tr,t,groups,rank)
            pp=[]
            for kind,target,eligible in [(1,target1,eligible1),(2,target2,eligible2)]:
                m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=6310+11*f+kind,verbose=False,allow_writing_files=False)
                assert not np.any(eligible&va)
                m.fit(X[eligible],target[eligible]);pp.append(m.predict_proba(X[va],thread_count=4)[:,1]);m.save_model(str(ROOT/f'{prefix}_event{kind}_{name}_fold{f}.cbm'))
            preds[va,0]=pp[0];preds[va,1]=pp[1];preds[va,2]=tp[va]
        print(prefix,'fold',f,'seconds',round(time.time()-start,1),flush=True)
    out=d.select('pair_id','hand_id','time').with_columns(pl.Series('primary',preds[:,0]),pl.Series('secondary',preds[:,1]),pl.Series('type_probability',preds[:,2]));parts=[]
    for _,g in out.group_by('pair_id',maintain_order=True):
        g=g.sort('time','hand_id');a=g['primary'].to_numpy();b=g['secondary'].to_numpy();parts.append(g.with_columns(pl.Series('score',inclusion(a,b)),pl.Series('uncapped',np.minimum(1,a+b))))
    pl.concat(parts).write_parquet(ROOT/f'{prefix}_oof.parquet');(ROOT/f'{prefix}_subtype_metrics.json').write_text(json.dumps(metrics,indent=2));(ROOT/f'{prefix}_columns.json').write_text(json.dumps({'event':cols,'type':typecols},indent=2));print(metrics,flush=True)
if __name__=='__main__':main()
