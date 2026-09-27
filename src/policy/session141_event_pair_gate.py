import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,time,hashlib
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from session140_event_pair_features import ROOT as SOURCE

ROOT=Path('artifacts/pair_session141_event_gate');OLD=Path('artifacts/evidence_session43_pair_interactions');C=pl.col
WINDOWS={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}
NAMES=['none','directed_transfer','soft_play','coordinated_isolation']
STATS=['sum','mean','max','top3sum','count05','count09','excess05']
COLS=['fold_log_count','fold_log_min_actor_count','fold_log_max_actor_count']+[f'{h}_{pool}_{s}' for h in ['direct','soft'] for pool in ['all','actor_min','actor_max'] for s in STATS]

def stats(p):
    return np.array([p.sum(),p.mean() if len(p) else 0,p.max(initial=0),np.sort(p)[-3:].sum(),(p>=.5).sum(),(p>=.9).sum(),np.maximum(p-.5,0).sum()],np.float64)

def aggregate(actor,values):
    counts=np.bincount(actor,minlength=2);parts=[np.log1p([len(actor),counts.min(),counts.max()])]
    for j in range(2):
        p=values[:,j];z=np.stack([stats(p[actor==r]) for r in range(2)])
        parts.extend([stats(p),z.min(0),z.max(0)])
    return np.concatenate(parts).astype(np.float32)

def prepare():
    ROOT.mkdir(exist_ok=True);assert (SOURCE/'audit.json').exists()
    d=pl.read_parquet(OLD/'window_features.parquet');actions=pl.read_parquet(sorted(SOURCE.glob('T*.parquet')))
    d.write_parquet(ROOT/'frame.parquet');features=np.zeros((4,len(d),len(COLS)),np.float32)
    row={(r['window'],r['pair_id']):i for i,r in enumerate(d.select('window','pair_id').to_dicts())};checks=0
    for w,(lo,hi) in WINDOWS.items():
        q=actions.filter((C('time_index')>=lo)&(C('time_index')<hi))
        for (pid,),g in q.group_by('pair_id'):
            if (w,pid) not in row:continue
            i=row[w,pid];actor=g['actor'].to_numpy()
            for f in range(4):
                values=g.select(f'direct_outer{f}',f'soft_outer{f}').to_numpy();v=aggregate(actor,values);features[f,i]=v
                np.testing.assert_allclose(v,aggregate(1-actor,values),atol=0,rtol=0)
                np.testing.assert_allclose(v,aggregate(actor[::-1],values[::-1]),atol=1e-5,rtol=1e-6);checks+=1
    assert features.shape[-1]==len(COLS)==45 and np.isfinite(features).all()
    np.save(ROOT/'features.npy',features)
    sources=json.load(open(SOURCE/'audit.json'));folds=json.load(open('artifacts/policy/table_folds.json'));exclusions=0
    for record in sources:
        for ref in record['models']:
            p=Path(ref['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
            if ref['outer']==record['native_fold']:continue
            meta=json.load(open(p.parent.with_suffix('.json')))
            for fam in meta['records']:
                assert record['table'] not in fam['training_tables']
                assert all(folds[t]!=ref['outer'] for t in fam['training_tables'])
            exclusions+=1
    cfg=dict(method=__doc__,columns=COLS,base_columns=json.load(open(OLD/'config.json'))['base_columns'],source=str(SOURCE),
             model='CatBoost600 depth5 lr.04 L2=10 MultiClass seed991+outer',weights='full1, short.5',
             target='trusted pair family; no unknown targets',raw_feature_exclusion_checks=exclusions,role_and_row_checks=checks,
             feature_sha256=hashlib.sha256((ROOT/'features.npy').read_bytes()).hexdigest(),
             caveat='Initial known-label gate only. Full-phase source event features persist into cropped views; a positive result requires strict raw-window rebuilding and query-complete population validation.')
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2));print('PREPARED',features.shape,exclusions,checks,flush=True)

def train():
    cfg=json.load(open(ROOT/'config.json'));d=pl.read_parquet(ROOT/'frame.parquet');extra=np.load(ROOT/'features.npy');bc=cfg['base_columns'];base=d.select(bc).to_numpy();fv=d['fold'].to_numpy();y=np.array([NAMES.index(v) for v in d['behavior_family']]);wt=np.where(d['window'].to_numpy()=='full',1,.5)
    probs={n:np.zeros((len(d),4)) for n in ['control','event']};audit=[];start=time.time()
    archived=pl.read_parquet(OLD/'oof.parquet').select('pair_id','window','residual_only')
    for f in range(4):
        tr=fv!=f;va=fv==f;training_tables=set(d.filter(pl.Series(tr))['table_id']);valid_tables=set(d.filter(pl.Series(va))['table_id']);assert not training_tables&valid_tables
        old=CatBoostClassifier();old.load_model(str(OLD/f'residual_only_fold{f}.cbm'));probs['control'][va]=old.predict_proba(base[va],thread_count=2)
        check=d.filter(pl.Series(va)).select('pair_id','window').with_columns(pl.Series('replay',1-probs['control'][va,0])).join(archived,on=['pair_id','window'],validate='1:1');err=float(abs(check['replay']-check['residual_only']).max());assert err==0
        x=np.column_stack([base,extra[f]]);m=CatBoostClassifier(iterations=600,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=2,random_seed=991+f,verbose=False,allow_writing_files=False)
        m.fit(x[tr],y[tr],sample_weight=wt[tr]);p=m.predict_proba(x[va],thread_count=2);path=ROOT/f'event_fold{f}.cbm';m.save_model(str(path));rr=CatBoostClassifier();rr.load_model(str(path));np.testing.assert_array_equal(p,rr.predict_proba(x[va],thread_count=2));np.testing.assert_array_equal(p,rr.predict_proba(x[va][::-1],thread_count=2)[::-1]);probs['event'][va]=p
        audit.append(dict(fold=f,training_tables=sorted(training_tables),validation_tables=sorted(valid_tables),training_rows=int(tr.sum()),validation_rows=int(va.sum()),control_replay_error=err,new_model_replay_exact=True,query_reversal_exact=True,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('TRAINED',f,time.time()-start,flush=True)
    out=d.select('pair_id','table_id','fold','window','label','behavior_family','n_ev_in').with_columns(*[pl.Series(n,1-v[:,0]) for n,v in probs.items()]);out.write_parquet(ROOT/'oof.parquet');(ROOT/'fit_audit.json').write_text(json.dumps(audit,indent=2))

def compare():
    d=pl.read_parquet(ROOT/'oof.parquet');reports=[]
    for w in WINDOWS:
        q=d.filter(C('window')==w);current=pl.read_parquet(f'artifacts/evidence_session91_family_uncertainty/{w}.parquet').select('pair_id',C('risk_score').alias('r33'));q=q.join(current,on='pair_id',validate='1:1');assert len(q)==d.filter(C('window')==w).height
        y=q['label'].to_numpy();weights=np.where(y==1,1,50);r=dict(window=w,n=len(q),positive=int(y.sum()),metrics={})
        for n in ['r33','control','event']:
            s=q[n].to_numpy();r['metrics'][n]=dict(known_AP=average_precision_score(y,s),negative_weight50_AP=average_precision_score(y,s,sample_weight=weights),fold_weight50_AP=[average_precision_score(z['label'],z[n],sample_weight=np.where(z['label'].to_numpy()==1,1,50)) for _,z in q.sort('fold').group_by('fold',maintain_order=True)],positives_below05=int(((y==1)&(s<.5)).sum()),negatives_above05=int(((y==0)&(s>=.5)).sum()))
                                                                              
        tables=q['table_id'].unique().sort().to_list();idx=np.array([tables.index(t) for t in q['table_id']]);rng=np.random.default_rng(14101);boots=[]
        for _ in range(2000):
            counts=np.bincount(rng.integers(len(tables),size=len(tables)),minlength=len(tables));sw=weights*counts[idx]
            if (sw*y).sum()==0 or (sw*(1-y)).sum()==0:continue
            boots.append(average_precision_score(y,q['event'],sample_weight=sw)-average_precision_score(y,q['control'],sample_weight=sw))
        r['event_minus_control_weight50']=r['metrics']['event']['negative_weight50_AP']-r['metrics']['control']['negative_weight50_AP'];r['pool_CI95_fixed_predictions']=np.quantile(boots,[.025,.975]).tolist();reports.append(r)
    (ROOT/'comparison.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'compare':compare}[sys.argv[1]]()
