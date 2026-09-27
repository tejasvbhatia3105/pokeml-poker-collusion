\
\
\
\
\
\
\
import gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
import session98_tabicl_events as tab
from session57_isolation_pressure import data, targets, assemble

ROOT = Path('artifacts/evidence_session100_isolation_bags')
C = pl.col
KINDS = ['cat_full','cat128','tabicl']

def inputs():
    full,d,a,ac = data()
    ex = np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x']
    assert len(ex)==len(a)
    ax = np.column_stack([a.select(ac).to_numpy(),ex]).astype(np.float32)
    hc = json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event']
    g = a['row'].to_numpy()
    count = np.bincount(g,minlength=len(d))
    support = count>0
    total = np.zeros((len(d),ax.shape[1]),np.float64)
    lo = np.full_like(total,np.inf)
    hi = np.full_like(total,-np.inf)
    np.add.at(total,g,ax)
    np.minimum.at(lo,g,ax)
    np.maximum.at(hi,g,ax)
    mean = total/np.maximum(count,1)[:,None]
    for v in [lo,hi,mean]:v[~support]=-2
    x = np.column_stack([d.select(hc).to_numpy(),np.log1p(count),mean,lo,hi]).astype(np.float32)
    assert np.isfinite(x).all()
    return full,d,a,x,support,count,ax

def cat(f,k):
    return CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,
        l2_leaf_reg=8,random_seed=6311+11*f+k,thread_count=2,verbose=False,
        allow_writing_files=False)

def prepare():
    ROOT.mkdir(exist_ok=True)
    full,d,a,x,support,count,ax = inputs()
    cfg={'method':__doc__,'features':x.shape[1],'selected_features':128,
         'selection':'top full Cat outer-training importance, original order',
         'Cat':'400D5 lr.035 L2=8 seed6311+11fold+zero_based_head',
         'TabICL':tab.CONFIG['TabICL'],'checkpoint':str(tab.CHECKPOINT),
         'checkpoint_sha256':tab.SHA,'pooling':'mean,min,max,log1p(count)',
         'supported_hands':int(support.sum()),'truth_hands':int(d['evidence'].sum()),
         'unsupported_truth':int(((d['evidence'].to_numpy()==1)&~support).sum()),
         'targets':'same two original59 censored hand targets; no action pseudo labels',
         'other_families':'unchanged59','recipe':'fixed equal average with full62'}
    assert cfg['unsupported_truth']==0
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    fv=d['fold'].to_numpy();fm=full['behavior_family'].to_numpy()=='coordinated_isolation'
    records=[];start=time.time()
    for kind in KINDS:(ROOT/kind).mkdir(exist_ok=True)
    for f in range(4):
        p1,p2,e1,e2,_=targets(full,f,'coordinated_isolation')
        ys=np.column_stack([p1,p2])[fm];es=np.column_stack([e1,e2])[fm]
        for k in range(2):
            tr=es[:,k]&support;va=(fv==f)&support;y=ys[:,k]
            assert not(tr&va).any()
            m=cat(f,k);m.fit(x[tr],y[tr]);m.save_model(str(ROOT/'cat_full'/f'fold{f}_head{k}.cbm'))
            p=m.predict_proba(x[va],thread_count=2)[:,1]
            np.save(ROOT/'cat_full'/f'fold{f}_head{k}_query.npy',p)
            imp=m.get_feature_importance(thread_count=2)
            selected=np.sort(np.lexsort((np.arange(len(imp)),-imp))[:128])
            context=ROOT/f'fold{f}_head{k}_context.npz'
            np.savez_compressed(context,x_train=x[tr][:,selected],y_train=y[tr].astype(np.int8),
                                x_query=x[va][:,selected],training_rows=np.flatnonzero(tr),
                                query_rows=np.flatnonzero(va),selected_columns=selected)
            m=cat(f,k);m.fit(x[tr][:,selected],y[tr]);m.save_model(str(ROOT/'cat128'/f'fold{f}_head{k}.cbm'))
            np.save(ROOT/'cat128'/f'fold{f}_head{k}_query.npy',m.predict_proba(x[va][:,selected],thread_count=2)[:,1])
            records.append({'fold':f,'head':k,'training_hands':int(tr.sum()),
                            'positive_training_hands':int(y[tr].sum()),'query_hands':int(va.sum()),
                            'importance_retained':float(imp[selected].sum()),
                            'context_sha256':hashlib.file_digest(context.open('rb'),'sha256').hexdigest(),
                            'validation_overlap':0,'elapsed':time.time()-start})
            (ROOT/'fit_audit.json').write_text(json.dumps(records,indent=2))
            print(records[-1],flush=True)

def neural():
    import torch
    assert torch.backends.mps.is_available()
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.65)
    assert hashlib.file_digest(tab.CHECKPOINT.open('rb'),'sha256').hexdigest()==tab.SHA
    for f in range(4):
        for k in range(2):
            z=np.load(ROOT/f'fold{f}_head{k}_context.npz')
            m=tab.estimator(f)
            m.fit(z['x_train'],z['y_train'])
            p=m.predict_proba(z['x_query'])[:,1]
            assert np.isfinite(p).all()
            np.save(ROOT/'tabicl'/f'fold{f}_head{k}_query.npy',p)
            print('TABICL_DONE',f,k,flush=True)
            del m;gc.collect();torch.mps.empty_cache()

def combine():
    _,d,_,_,_,_,_=inputs()
    base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet')
    for kind in KINDS:
        pp=np.zeros((len(d),2))
        for f in range(4):
            for k in range(2):
                z=np.load(ROOT/f'fold{f}_head{k}_context.npz')
                pp[z['query_rows'],k]=np.load(ROOT/kind/f'fold{f}_head{k}_query.npy')
        q=d.select('pair_id','hand_id').with_columns(pl.Series('replacement1',pp[:,0]),pl.Series('replacement2',pp[:,1]))
        out=base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(
            pl.coalesce('replacement1','bg_primary').alias('bg_primary'),
            pl.coalesce('replacement2','bg_secondary').alias('bg_secondary')).drop('replacement1','replacement2')
        assert len(out)==len(base)
        out.write_parquet(ROOT/kind/'event_oof.parquet');assemble(ROOT/kind)

if __name__=='__main__':
    {'prepare':prepare,'neural':neural,'combine':combine}[sys.argv[1]]()
