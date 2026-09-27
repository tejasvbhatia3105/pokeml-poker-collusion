\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib,time
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session172_negative_action_em as baseline
s=baseline.s;negative=baseline.negative
ROOT=Path('artifacts/evidence_session180_cohort_conditioned_action_em');C=pl.col
CONFIG=dict(baseline.CONFIG,method=__doc__,categories=['ordinary_negative','ordinary_positive','primary_positive','secondary_positive'],
    inference='Normalize the last three categories for every query, without a query-label argument; then use the unchanged capped-list inclusion.')

def conditional(p):
    assert p.ndim==2 and p.shape[1]==4 and np.isfinite(p).all() and np.all(p>=0)
    total=p[:,1:].sum(1);assert np.all(total>0)
    return p[:,1:]/total[:,None]

def math_check():
    rng=np.random.default_rng(18001);q=rng.dirichlet([1,1,1],100);c=np.geomspace(1e-8,.999999,100)
    p=np.column_stack([1-c,c[:,None]*q]);out=conditional(p)
    np.testing.assert_allclose(out,q,atol=2e-16,rtol=1e-14)
    np.testing.assert_array_equal(out[::-1],conditional(p[::-1]))
    return float(abs(out-q).max())

def main(verify=False):
    ROOT.mkdir(exist_ok=True);error=math_check()
    assert json.load(open(baseline.ROOT/'verification.json'))['models']==24
    d,meta,x,bag,groups,aw=s.load();nm,nx,nw,chosen=baseline.controls();xx=np.concatenate([x,nx]);n=len(x)
    cfg=dict(CONFIG,positive_config=json.load(open(s.data.ROOT/'config.json')),negative_config=json.load(open(negative.ROOT/'config.json')),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),baseline_source_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),action_source_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest())
    cp=ROOT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:assert not verify;cp.write_text(json.dumps(cfg,indent=2))
    start=time.time();records=[];parts=[]
    for fold in range(4):
        tr=d['fold'].to_numpy()!=fold;va=~tr;ni=np.flatnonzero(nm['fold'].to_numpy()!=fold);negative_pairs=nm[ni]['pair_id'].n_unique()
        p=s.math.initialize(s.old.initial(d,groups),bag)
        mutation=d.with_columns(pl.when(pl.Series(va)).then(pl.lit(999)).otherwise(C('evidence_rank')).alias('evidence_rank'))
        for step in range(1,CONFIG['em_steps']+1):
            a=s.targets(d,groups,bag,p,tr,aw);b=s.targets(mutation,groups,bag,p,tr,aw)
            for u,v in zip(a[:4],b[:4]):np.testing.assert_array_equal(u,v)
            assert a[4:]==b[4:];rows,cat,w,used,minimum,excluded=a
            positive_pairs=d.filter(pl.Series(used))['pair_id'].n_unique();pw=w*(positive_pairs/w.sum());cw=nw[ni]*(positive_pairs/negative_pairs)
            np.testing.assert_allclose(pw.sum(),cw.sum(),rtol=1e-12)
            rows=np.r_[rows,n+ni];cat=np.r_[cat+1,np.zeros(len(ni),dtype=cat.dtype)];w=np.r_[pw,cw];w*=len(w)/w.sum()
            tables=sorted(set(meta.filter(C('fold')!=fold)['table_id'])|set(nm[ni]['table_id']))
            assert not set(tables)&(set(d.filter(C('fold')==fold)['table_id'])|set(nm.filter(C('fold')==fold)['table_id']))
            assert set(cat)=={0,1,2,3}
            rec=dict(fold=fold,step=step,minimum=minimum,positive_pairs=positive_pairs,negative_pairs=negative_pairs,negative_actions=len(ni),weighted_rows=len(rows),training_tables=tables,excluded_incompatible_pairs=excluded,
                target_sha256=s.old.checksum(rows,cat,w),negative_source_rows_sha256=hashlib.sha256(chosen[ni].tobytes()).hexdigest(),protected_rank_mutation_exact=True)
                                                                                  
            control=json.load(open(baseline.ROOT/f'fold{fold}_em{step}.json'));assert rec['negative_source_rows_sha256']==control['negative_source_rows_sha256']
            assert rec['positive_pairs']==control['positive_pairs'] and rec['negative_pairs']==control['negative_pairs']
            path=ROOT/f'fold{fold}_em{step}.cbm'
            if path.exists():
                saved=json.load(open(path.with_suffix('.json')));assert all(saved[k]==v for k,v in rec.items());assert saved['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest();m=CatBoostClassifier();m.load_model(str(path))
            else:
                assert not verify;m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='MultiClass',random_seed=12301+fold,thread_count=4,verbose=False,allow_writing_files=False)
                m.fit(xx[rows],cat,sample_weight=w);m.save_model(str(path));rec['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();path.with_suffix('.json').write_text(json.dumps(rec,indent=2))
            assert m.tree_count_==400 and np.array_equal(m.classes_,np.arange(4));raw=m.predict_proba(x,thread_count=2);p=conditional(raw)
            ix=np.flatnonzero(meta['fold'].to_numpy()==fold)[::97]
            np.testing.assert_array_equal(p[ix],conditional(m.predict_proba(x[ix[::-1]],thread_count=2))[::-1])
            records.append(rec);print('COHORT_ACTION_EM',fold,step,len(rows),round(time.time()-start,1),flush=True)
        q=s.output(d,groups,bag,p,fold,minimum);path=ROOT/f'fold{fold}.parquet'
        if verify:assert q.equals(pl.read_parquet(path))
        else:q.write_parquet(path)
        parts.append(q)
    if verify:
        assert pl.concat(parts).equals(pl.read_parquet(ROOT/'oof.parquet'))
        (ROOT/'verification.json').write_text(json.dumps(dict(models=len(records),final_prediction_error=0,pool_exclusions=True,protected_rank_mutations_exact=True,negative_pairs_confirmed=True,unknown_pairs_used=False,
            conditional_distribution_error=error,inference_branch='positive-cohort distribution for every query; no query-label input',records=records),indent=2))
    else:pl.concat(parts).write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(records,indent=2))
    print('COMPLETE',verify,len(records),round(time.time()-start,1),flush=True)

if __name__=='__main__':
    import sys
    assert sys.argv[1] in ['train','verify'];main(sys.argv[1]=='verify')
