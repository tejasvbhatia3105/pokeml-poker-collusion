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
import session123_family_holdout as old
import session162_action_em_math as math
import session163_generic_action_data as data

ROOT=Path('artifacts/evidence_session164_generic_action_em');C=pl.col
CONFIG=dict(method=__doc__,iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,em_steps=2,seed=12301,
    selection='Four fixed all-family folds; no held-out stopping, threshold tuning or teacher initialization.',
    hypothesis='Independent latent action categories; any primary action makes the hand primary, else any secondary makes it secondary.')

def load():
    d=pl.read_parquet(old.data.ROOT/'hands.parquet');meta=pl.read_parquet(data.ROOT/'actions.parquet');x=np.load(data.ROOT/'x.npy',mmap_mode='r')
    assert len(meta)==len(x)==159748 and np.array_equal(meta['action_row'],np.arange(len(meta)))
    bag=meta['hand_index'].to_numpy();groups=[g.sort('time_index','hand_id')['hand_index'].to_numpy() for _,g in d.group_by('pair_id')]
    groups.sort(key=lambda g:d['pair_id'][int(g[0])]);counts=np.bincount(bag,minlength=len(d));assert (counts>0).all()
    paircounts=np.zeros(len(d))
    for ix in groups:paircounts[ix]=counts[ix].sum()
    return d,meta,x,bag,groups,1/paircounts[bag]

def targets(d,groups,bag,p,tr,action_weight):
    hp=math.hand_probabilities(p,bag,len(d))[0];ranks=d['evidence_rank'].fill_null(0).to_numpy();ht=np.zeros_like(hp);used=np.zeros(len(d),bool);counts=[];excluded=0
    for ix in groups:
        if not tr[ix[0]]:continue
        e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])]
        pp,_,_=old.posterior(hp[ix],e)
        if pp is None:excluded+=1;continue
        ht[ix]=pp;used[ix]=True;counts.append(len(e))
    assert not np.any(used&~tr)
    at=math.responsibilities(p,bag,ht);rows,category=np.nonzero((at>1e-7)&used[bag,None]);w=at[rows,category]*action_weight[rows];w*=len(rows)/w.sum()
    return rows,category,w,used,min(counts),excluded

def output(d,groups,bag,p,fold,minimum):
    hp=math.hand_probabilities(p,bag,len(d))[0];parts=[]
    for ix in groups:
        if d['fold'][int(ix[0])]!=fold:continue
        parts.append(d[ix].select('hand_index','pair_id','hand_id','table_id','fold','behavior_family','evidence').with_columns(pl.Series('score',old.conditioned(hp[ix,1:],minimum)),pl.Series('primary',hp[ix,1]),pl.Series('secondary',hp[ix,2])))
    return pl.concat(parts).sort('hand_index')

def run(verify=False):
    ROOT.mkdir(exist_ok=True);assert (math.ROOT/'verification.json').exists()
    d,meta,x,bag,groups,aw=load();cfg=dict(CONFIG,data_config=json.load(open(data.ROOT/'config.json')),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),math_sha256=hashlib.sha256(Path(math.__file__).read_bytes()).hexdigest())
    assert cfg['data_config']['x_sha256']==hashlib.sha256((data.ROOT/'x.npy').read_bytes()).hexdigest()
    path=ROOT/'config.json'
    if path.exists():assert json.loads(path.read_text())==cfg
    else:assert not verify;path.write_text(json.dumps(cfg,indent=2))
    records=[];start=time.time();maxerror=0.
    for fold in range(4):
        tr=d['fold'].to_numpy()!=fold;va=~tr;hp=old.initial(d,groups);p=math.initialize(hp,bag)
        np.testing.assert_allclose(math.hand_probabilities(p,bag,len(d))[0],hp,atol=1e-10,rtol=0)
        mutation=d.with_columns(pl.when(pl.Series(va)).then(pl.lit(999)).otherwise(C('evidence_rank')).alias('evidence_rank'))
        for step in range(CONFIG['em_steps']):
            a=targets(d,groups,bag,p,tr,aw);b=targets(mutation,groups,bag,p,tr,aw)
            for one,two in zip(a[:4],b[:4]):np.testing.assert_array_equal(one,two)
            assert a[4:]==b[4:];rows,category,w,used,minimum,excluded=a
            assert not np.any(meta['fold'].to_numpy()[rows]==fold)
            tables=sorted(meta[rows]['table_id'].unique().to_list());assert not set(tables)&set(d.filter(C('fold')==fold)['table_id'])
            record=dict(fold=fold,step=step+1,training_tables=tables,weighted_rows=len(rows),minimum=minimum,
                target_sha256=old.checksum(rows,category,w),excluded_incompatible_pairs=excluded,protected_rank_mutation_exact=True)
            path=ROOT/f'fold{fold}_em{step+1}.cbm'
            if path.exists():
                saved=json.load(open(path.with_suffix('.json')));assert all(saved[k]==v for k,v in record.items())
                assert saved['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                m=CatBoostClassifier();m.load_model(str(path))
            else:
                assert not verify
                m=CatBoostClassifier(iterations=CONFIG['iterations'],depth=CONFIG['depth'],learning_rate=CONFIG['learning_rate'],l2_leaf_reg=CONFIG['l2_leaf_reg'],loss_function='MultiClass',random_seed=CONFIG['seed']+fold,thread_count=4,verbose=False,allow_writing_files=False)
                m.fit(x[rows],category,sample_weight=w);m.save_model(str(path));record['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();path.with_suffix('.json').write_text(json.dumps(record,indent=2))
            assert m.tree_count_==400 and np.array_equal(m.classes_,np.arange(3))
            p=m.predict_proba(x,thread_count=2)
            sample=np.flatnonzero(meta['fold'].to_numpy()==fold)[::97]
            np.testing.assert_array_equal(p[sample],m.predict_proba(x[sample[::-1]],thread_count=2)[::-1])
            records.append(record);print('ACTION_EM',fold,step+1,len(rows),round(time.time()-start,1),flush=True)
        q=output(d,groups,bag,p,fold,minimum);path=ROOT/f'fold{fold}.parquet'
        if verify:
            saved=pl.read_parquet(path).sort('hand_index');assert q.select('pair_id','hand_id').equals(saved.select('pair_id','hand_id'))
            err=float(abs(q.select('score','primary','secondary').to_numpy()-saved.select('score','primary','secondary').to_numpy()).max());assert err==0;maxerror=max(maxerror,err)
        else:q.write_parquet(path)
    if verify:
        (ROOT/'verification.json').write_text(json.dumps(dict(models=len(records),scope='all-family control only',records=records,final_prediction_error=maxerror,pool_exclusions=True,query_row_order_reversal_exact=True,limitations='Independent-action model is a hypothesis; mathematical enumeration does not establish the data generator. Reused public labels, no held-family result or candidate.'),indent=2))
    else:
        pl.concat([pl.read_parquet(ROOT/f'fold{f}.parquet') for f in range(4)]).write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(records,indent=2))
    print('COMPLETE',verify,len(records),round(time.time()-start,1),flush=True)

if __name__=='__main__':
    import sys
    assert sys.argv[1] in ['train','verify'];run(sys.argv[1]=='verify')
