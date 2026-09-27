\
\
\
\
import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');sys.path.insert(0,'src/policy')
from pathlib import Path
import json,time
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from evidence_data import load
OUT=Path('artifacts/evidence_session4');root=Path('artifacts/policy');C=pl.col
def action_features(x,act,lag,p1,p2,direction):
                                                          
    a1=act==p1[:,None];a2=act==p2[:,None];l1=lag==p1[:,None];l2=lag==p2[:,None]
    lower=np.where(direction[:,None]<0,a1,a2);higher=np.where(direction[:,None]>0,a1,a2)
    ll=np.where(direction[:,None]<0,l1,l2);lh=np.where(direction[:,None]>0,l1,l2)
    tie=direction[:,None]==0
    lower=np.where(tie,a1|a2,lower);higher=np.where(tie,a1|a2,higher)
    ll=np.where(tie,l1|l2,ll);lh=np.where(tie,l1|l2,lh)
    valid=act>=0;other=valid&~(a1|a2)
    same_prev=np.zeros_like(valid);same_prev[:,1:]=(act[:,1:]==act[:,:-1])&valid[:,1:]&valid[:,:-1]
    role=np.stack([valid,lower,higher,other,ll,lh,(lag>=0)&~(l1|l2),same_prev],2)
    return np.concatenate([np.where(valid[:,:,None],x,-1),role],2).reshape(len(x),-1).astype('float32')
def build():
    extra=Path(json.loads(Path('artifacts/seq_v6/config.json').read_text())['tokens3']);v1=extra.parent/'seq_tokens';ad=extra.parent/'action_tokens'
    ix=pl.read_parquet(OUT/'hand_index.parquet');parts=[];t=time.time();checked=False
    for (table,),q in ix.group_by('table_id'):
        with np.load(v1/f'{table}.npz') as z:
            pi=np.repeat(np.arange(len(z['pair_id'])),z['n']);meta=pl.DataFrame({'row':np.arange(len(z['hand_id'])),'pair_index':pi,'hand_id':z['hand_id'],'pair_id':np.repeat(z['pair_id'],z['n'])})
            m=q.join(meta,on=['pair_id','hand_id'],validate='1:1')
        assert len(m)==len(q)
        with np.load(ad/f'{table}.npz') as a:
            h=a['hidx'][m['row'].to_numpy()];p1=a['p1'][m['pair_index'].to_numpy()];p2=a['p2'][m['pair_index'].to_numpy()]
            raw=np.asarray(np.load(ad/f'{table}_XA.npy',mmap_mode='r')[h],dtype='float32');act=a['ACT'][h];lag=a['LAG'][h]
        dr=m['net_direction'].to_numpy();f=action_features(raw,act,lag,p1,p2,dr)
        if not checked:
            assert np.array_equal(f,action_features(raw,act,lag,p2,p1,-dr));checked=True
        parts.append(m.select('pair_id','hand_id').hstack(pl.DataFrame(f,schema=[f'raw_action_{i}' for i in range(f.shape[1])])) )
    pl.concat(parts).write_parquet(OUT/'raw_action_features.parquet');print('raw action cache',round(time.time()-t,1),'endpoint swap passed',flush=True)
def main():
    if not (OUT/'raw_action_features.parquet').exists():build()
    d,_=load();extra=pl.read_parquet(list((root/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((root/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(extra,on=['pair_id','hand_id']).join(pl.read_parquet(OUT/'raw_action_features.parquet'),on=['pair_id','hand_id'])
    d=d.join(pl.read_parquet(OUT/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
    old=json.loads((root/'relationship_evidence/columns.json').read_text());new=[c for c in d.columns if c.startswith('raw_action_')]
    y=d['evidence'].to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();t=time.time()
    for mode,cols in [('raw_only',new+['relative_time']),('raw_combined',old+new)]:
        path=OUT/f'{mode}_oof.parquet'
        if path.exists():continue
        X=d.select(cols).to_numpy();parts=[]
        for f in range(4):
            pr=np.full(len(d),np.nan)
            for b in ['directed_transfer','soft_play','coordinated_isolation']:
                tr=(fv!=f)&(fam==b);va=(fv==f)&(fam==b)
                m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,loss_function='Logloss',l2_leaf_reg=8,
                    thread_count=4,random_seed=1710+11*f,verbose=False,allow_writing_files=False)
                m.fit(X[tr],y[tr]);pr[va]=m.predict_proba(X[va])[:,1];m.save_model(str(OUT/f'{mode}_{b}_fold{f}.cbm'))
            va=fv==f;parts.append(d.filter(pl.Series(va)).select('pair_id','hand_id','behavior_family','evidence','fold').with_columns(pl.Series('score',pr[va])))
            print(mode,'fold',f,round(time.time()-t,1),flush=True)
        pl.concat(parts).write_parquet(path);(OUT/f'{mode}_columns.json').write_text(json.dumps(cols))
if __name__=='__main__':main()
