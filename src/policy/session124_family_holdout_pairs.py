\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,time,hashlib
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
import session119_private_partner_data as source
from session122_family_blind_hands import aggregate,GROUPS
ROOT=Path('artifacts/pair_session124_family_holdout');C=pl.col
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
CONFIG=dict(method=__doc__,iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,seed=12401,
    weighting='Unweighted trusted relationships; no unknown labels or pseudo targets',selection='Fixed model/schedule; no held-out stopping')

def prepare():
    ROOT.mkdir(exist_ok=True);bags=pl.read_parquet('artifacts/evidence_session115_relationship_data/bags.parquet').sort('pair_id').with_row_index('pair_index')
    frames=[];arrays=[]
    for a in json.load(open(source.ROOT/'audit.json'))['tables']:
        table=a['table'];meta=pl.read_parquet(source.ROOT/f'{table}.parquet');x=np.load(source.ROOT/f'{table}.npz')['x']
        z=meta.join(bags.select('pair_id','pair_index'),on='pair_id',validate='m:1',maintain_order='left');assert len(z)==len(x)
        frames.append(z);arrays.append(x)
    meta=pl.concat(frames);raw=np.concatenate(arrays);b=meta['pair_index'].to_numpy();cls=meta['action_class'].to_numpy();post=raw[:,source.PUBLIC.index('street_no')]>0
    masks=[np.ones(len(raw),bool),cls==0,(cls==0)&post,cls==2,(cls==2)&post,cls==3,(cls==3)&post];cols=[];blocks=[];checks=0
    for name,mask in zip(GROUPS,masks):
        x,count=aggregate(raw,b,mask,len(bags));blocks.append(x)
        cols+=[name+'_log_count']+[name+'_'+stat+'_'+c for stat in ['mean','min','max'] for c in source.PUBLIC+source.PRIVATE]
        for i in np.random.default_rng(12401).choice(len(bags),32,replace=False):
            z=raw[mask&(b==i)];expected=np.r_[np.log1p(len(z)),z.mean(0,dtype=np.float64),z.min(0),z.max(0)] if len(z) else np.r_[0,np.full(3*raw.shape[1],-2)]
            np.testing.assert_allclose(x[i],expected,rtol=1e-6,atol=1e-5);checks+=1
    blocks.append(np.log1p(bags['n_hands'].to_numpy())[:,None]);cols.append('log_shared_hands')
    x=np.column_stack(blocks).astype(np.float32);assert np.isfinite(x).all() and len(bags)==1860
    np.save(ROOT/'x.npy',x);bags.write_parquet(ROOT/'pairs.parquet')
    (ROOT/'data_config.json').write_text(json.dumps(dict(columns=cols,source_config=json.load(open(source.ROOT/'config.json')),
        x_sha256=hashlib.sha256((ROOT/'x.npy').read_bytes()).hexdigest(),scalar_group_checks=checks),indent=2))
    print('pair input',x.shape,flush=True)

def train():
    if not (ROOT/'x.npy').exists():prepare()
    x=np.load(ROOT/'x.npy');d=pl.read_parquet(ROOT/'pairs.parquet');y=d['label'].to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy()
    cfg=dict(CONFIG,data=json.load(open(ROOT/'data_config.json')),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cp=ROOT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:cp.write_text(json.dumps(cfg,indent=2))
    rows=[];audit=[];start=time.time()
    for held in ['all']+FAMILIES:
        for f in range(4):
            tr=fv!=f;va=fv==f
            if held!='all':tr&=fam!=held;va&=(fam==held)|(y==0)
            path=ROOT/f'{held}_fold{f}.cbm'
            if path.exists():m=CatBoostClassifier();m.load_model(str(path))
            else:
                m=CatBoostClassifier(iterations=CONFIG['iterations'],depth=CONFIG['depth'],learning_rate=CONFIG['learning_rate'],
                    l2_leaf_reg=CONFIG['l2_leaf_reg'],random_seed=CONFIG['seed']+f,thread_count=2,verbose=False,allow_writing_files=False)
                m.fit(x[tr],y[tr]);m.save_model(str(path))
            p=m.predict_proba(x[va],thread_count=2)[:,1]
            rows.append(d.filter(pl.Series(va)).select('pair_id','table_id','fold','label','behavior_family').with_columns(pl.lit(held).alias('held_family'),pl.Series('risk',p)))
            record=dict(held_family=held,fold=f,training_pair_ids=d.filter(pl.Series(tr))['pair_id'].to_list(),
                train_tables=sorted(set(d['table_id'].to_numpy()[tr])),training_positives=int(y[tr].sum()),training_negatives=int((y[tr]==0).sum()),
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            assert not np.any(tr&va) and not set(record['train_tables'])&set(d['table_id'].to_numpy()[va])
            if held!='all':assert not np.any(fam[tr]==held)
            path.with_suffix('.json').write_text(json.dumps(record,indent=2));audit.append(record)
            print('PAIR_FAMILY_HOLDOUT',held,f,round(time.time()-start,1),flush=True)
    pl.concat(rows).write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));evaluate()

def evaluate():
    q=pl.read_parquet(ROOT/'oof.parquet');known=q.filter(C('held_family')=='all').select('pair_id',C('risk').alias('known_family_risk'));report=[]
    for held in FAMILIES:
        z=q.filter(C('held_family')==held).join(known,on='pair_id',validate='1:1');y=z['label'].to_numpy()
        for kind,col in [('known_family_control','known_family_risk'),('withheld_family','risk')]:
            p=z[col].to_numpy();n=y.sum();rank=np.argsort(-p,kind='stable');report.append(dict(family=held,kind=kind,positive_pairs=int(n),negative_pairs=int((y==0).sum()),
                AP=average_precision_score(y,p),AUC=roc_auc_score(y,p),recall_at_positive_count=float(y[rank[:n]].mean())))
    (ROOT/'comparison.json').write_text(json.dumps(dict(results=report,limitations='Trusted labelled subset only; excludes unknown backgrounds and differs from competition class prevalence. Same generic learner; current R33 is not the matched control. Historical feature design knew all families. Not a measured hidden-family score.'),indent=2));print(json.dumps(report,indent=2))

def verify():
    d=pl.read_parquet(ROOT/'pairs.parquet');x=np.load(ROOT/'x.npy');q=pl.read_parquet(ROOT/'oof.parquet');y=d['label'].to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();records=[]
    assert json.load(open(ROOT/'data_config.json'))['x_sha256']==hashlib.sha256((ROOT/'x.npy').read_bytes()).hexdigest()
    for rec in json.load(open(ROOT/'audit.json')):
        held=rec['held_family'];f=rec['fold'];tr=fv!=f;va=fv==f
        if held!='all':tr&=fam!=held;va&=(fam==held)|(y==0)
        assert rec['training_pair_ids']==d.filter(pl.Series(tr))['pair_id'].to_list()
        assert not set(rec['train_tables'])&set(d.filter(C('fold')==f)['table_id'])
        mutant=y.copy();mutant[~tr]=1-mutant[~tr];assert np.array_equal(y[tr],mutant[tr])
        path=ROOT/f'{held}_fold{f}.cbm';assert rec['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        m=CatBoostClassifier();m.load_model(str(path));p=m.predict_proba(x[va],thread_count=2)[:,1];ids=d.filter(pl.Series(va)).select('pair_id')
        z=ids.join(q.filter((C('held_family')==held)&(C('fold')==f)),on='pair_id',validate='1:1',maintain_order='left')
        assert np.array_equal(p,z['risk'].to_numpy()) and np.array_equal(p,m.predict_proba(x[va][::-1],thread_count=2)[:,1][::-1])
        records.append(dict(family=held,fold=f,replay_error=0,excluded_label_mutation_exact=True))
    (ROOT/'verification.json').write_text(json.dumps(dict(models=len(records),records=records),indent=2));print('verified',len(records))

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'verify':verify,'evaluate':evaluate}[sys.argv[1]]()
