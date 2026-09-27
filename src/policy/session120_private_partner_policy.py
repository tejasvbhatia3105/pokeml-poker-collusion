\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import itertools,json,time,hashlib
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
import session119_private_partner_data as data
ROOT=Path('artifacts/evidence_session120_private_partner_policy');C=pl.col
CONFIG=dict(method=__doc__,iterations=400,depth=6,learning_rate=.04,l2_leaf_reg=10,seed=12001,
    max_training_actions_per_condition=50000,weighting='equal total weight per represented training pair',
    exclusions='Six models per arm; each excludes two whole pool folds',
    conditions=[0,1],arms=['public','private'],selection='Fixed schedule; no heldout stopping or family splice')

def load():
    audit=json.load(open(data.ROOT/'audit.json'));metas=[];arrays=[]
    for a in audit['tables']:
        table=a['table'];m=pl.read_parquet(data.ROOT/f'{table}.parquet');x=np.load(data.ROOT/f'{table}.npz')['x']
        assert len(m)==len(x)==a['action_rows'];metas.append(m);arrays.append(x)
    return pl.concat(metas).with_row_index('action_row'),np.concatenate(arrays)

def design(x,kind,condition):
    xx=x[:,:len(data.PUBLIC)] if kind=='public' else x
    v=np.full(len(x),condition,np.float32) if np.isscalar(condition) else condition
    return np.column_stack([xx,v]).astype(np.float32)

def training(meta,f,h):
    fv=meta['fold'].to_numpy();lab=meta['label'].to_numpy();rng=np.random.default_rng(CONFIG['seed']+f*4+h)
    pieces=[]
    for condition in [0,1]:
        ix=np.flatnonzero(~np.isin(fv,[f,h])&(lab==condition))
        if len(ix)>CONFIG['max_training_actions_per_condition']:ix=np.sort(rng.choice(ix,CONFIG['max_training_actions_per_condition'],replace=False))
        pieces.append(ix)
    train=np.sort(np.concatenate(pieces));valid=np.flatnonzero(np.isin(fv,[f,h]));assert not np.intersect1d(train,valid).size
    pid=meta['pair_id'].to_numpy()[train];_,inverse,counts=np.unique(pid,return_inverse=True,return_counts=True)
    weights=1/counts[inverse];weights/=weights.mean()
    return train,valid,weights

def predict(model,x,kind):
    return np.stack([model.predict_proba(design(x,kind,k),thread_count=3) for k in [0,1]],1)

def main():
    ROOT.mkdir(exist_ok=True);meta,x=load();assert np.isfinite(x).all();meta.write_parquet(ROOT/'action_index.parquet')
    cfg=dict(CONFIG,data_config=json.load(open(data.ROOT/'config.json')),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cp=ROOT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:cp.write_text(json.dumps(cfg,indent=2))
    start=time.time();audit=[]
    for f,h in itertools.combinations(range(4),2):
        train,valid,w=training(meta,f,h);z=meta[valid];trmeta=meta[train]
        assert not set(trmeta['table_id'])&set(z['table_id'])
        for kind in CONFIG['arms']:
            path=ROOT/f'{kind}_exclude{f}{h}.cbm';out=ROOT/f'{kind}_exclude{f}{h}.parquet'
            if path.exists() and out.exists():continue
            model=CatBoostClassifier(iterations=CONFIG['iterations'],depth=CONFIG['depth'],learning_rate=CONFIG['learning_rate'],
                l2_leaf_reg=CONFIG['l2_leaf_reg'],loss_function='MultiClass',random_seed=CONFIG['seed']+4*f+h,
                thread_count=4,verbose=False,allow_writing_files=False)
            model.fit(design(x[train],kind,trmeta['label'].to_numpy()),trmeta['action_class'].to_numpy(),sample_weight=w)
            model.save_model(str(path));p=predict(model,x[valid],kind);assert np.array_equal(model.classes_,np.arange(4))
            z.select('action_row','pair_id','hand_id','action_no','actor','fold').with_columns(*[
                pl.Series(f'condition{c}_class{k}',p[:,c,k]) for c in [0,1] for k in range(4)]).write_parquet(out)
            record=dict(excluded_folds=[f,h],kind=kind,train_actions=len(train),validation_actions=len(valid),
                train_pair_ids=sorted(set(trmeta['pair_id'])),train_tables=sorted(set(trmeta['table_id'])),
                train_rows_sha256=hashlib.sha256(train.tobytes()).hexdigest(),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (ROOT/f'{kind}_exclude{f}{h}.json').write_text(json.dumps(record,indent=2));audit.append(record)
            print('PRIVATE_POLICY',f,h,kind,len(train),len(valid),round(time.time()-start,1),flush=True)
    print('complete',time.time()-start,flush=True)

def evaluate():
    meta,x=load();lab=meta['label'].to_numpy();action=meta['action_class'].to_numpy();probs={k:np.zeros((len(meta),2,4)) for k in CONFIG['arms']};seen=np.zeros(len(meta),int);checks=[]
    for f,h in itertools.combinations(range(4),2):
        train,valid,w=training(meta,f,h);seen[valid]+=1
        for kind in CONFIG['arms']:
            path=ROOT/f'{kind}_exclude{f}{h}.cbm';model=CatBoostClassifier();model.load_model(str(path));p=predict(model,x[valid],kind)
            saved=pl.read_parquet(ROOT/f'{kind}_exclude{f}{h}.parquet');assert np.array_equal(saved['action_row'],valid)
            expected=saved.select([f'condition{c}_class{k}' for c in [0,1] for k in range(4)]).to_numpy().reshape(-1,2,4)
            err=float(abs(p-expected).max());assert err==0
            record=json.load(open(ROOT/f'{kind}_exclude{f}{h}.json'));assert record['train_rows_sha256']==hashlib.sha256(train.tobytes()).hexdigest()
            assert not set(record['train_tables'])&set(meta[valid]['table_id'])
            assert record['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                                                                             
                                                                              
            probs[kind][valid]+=p/3;checks.append(dict(excluded_folds=[f,h],kind=kind,probability_replay_error=err))
    assert np.all(seen==3);rows=np.arange(len(meta));cols={}
    for kind,p in probs.items():
        log=np.log(np.clip(p[rows,:,action],1e-9,1));cols[kind+'_loglikelihood']=log[rows,lab]
        cols[kind+'_coordination_ratio']=log[:,1]-log[:,0]
    cols['private_gain']=cols['private_loglikelihood']-cols['public_loglikelihood']
    cols['private_coordination_contrast']=cols['private_coordination_ratio']-cols['public_coordination_ratio']
    q=meta.with_columns(*[pl.Series(k,v) for k,v in cols.items()]);q.write_parquet(ROOT/'action_scores.parquet')
    pair=q.group_by('pair_id').agg(C('table_id').first(),C('fold').first(),C('label').first(),C('family').first(),
        C(list(cols)).mean(),pl.len().alias('actions'));pair.write_parquet(ROOT/'pair_diagnostics.parquet')
    stats=[]
    for condition in [0,1]:
        p=pair.filter(C('label')==condition);pool=p.group_by('table_id').agg(C('private_gain').sum(),pl.len().alias('n')).sort('table_id')
        ix=np.random.default_rng(12001).integers(0,len(pool),(5000,len(pool)));boot=pool['private_gain'].to_numpy()[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
        stats.append(dict(condition=condition,pairs=len(p),public_nll=-p['public_loglikelihood'].mean(),private_nll=-p['private_loglikelihood'].mean(),
            gain=p['private_gain'].mean(),ci95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist(),
            fold_gains=p.group_by('fold').agg(C('private_gain').mean()).sort('fold')['private_gain'].to_list()))
    report=dict(conditions=stats,positive_family_gains=dict(pair.filter(C('label')==1).group_by('family').agg(C('private_gain').mean()).iter_rows()),
        action_rows=len(meta),checkpoints=checks,caveat='Action prediction diagnostic, not evidence MAP or causal identification. Private information can proxy omitted public state; ordinary-pair control is essential. No evidence model/candidate has been built.')
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    import sys
    {'train':main,'evaluate':evaluate}[sys.argv[1]]()
