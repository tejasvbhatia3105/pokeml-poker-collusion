\
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
from session7_likelihood import posterior
from session8_count_conditioning import conditioned
import session122_family_blind_hands as data
ROOT=Path('artifacts/evidence_session123_family_holdout');C=pl.col
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
CONFIG=dict(method=__doc__,iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,em_steps=2,seed=12301,
    initialization='Constant expected 2.5 primary and 2.5 secondary events per relationship; no fitted teacher',
    weighting='Equal total posterior weight per compatible training relationship',
    inference='Two-tier inclusion conditional on minimum listed count among training relationships; no family-specific minimum',
    selection='Fixed schedule; no held-out stopping; four pool folds; all-family control and three withheld-family arms')

def load():
    d=pl.read_parquet(data.ROOT/'hands.parquet');x=np.load(data.ROOT/'x.npy',mmap_mode='r')
    assert np.array_equal(d['hand_index'],np.arange(len(d)))
    groups=[g.sort('time_index','hand_id')['hand_index'].to_numpy() for _,g in d.group_by('pair_id')]
    groups.sort(key=lambda g:d['pair_id'][int(g[0])]);return d,x,groups

def masks(d,fold,held):
    fv=d['fold'].to_numpy();family=d['behavior_family'].to_numpy()
    tr=fv!=fold;va=fv==fold
    if held!='all':tr&=family!=held;va&=family==held
    assert not np.any(tr&va)
    return tr,va

def initial(d,groups):
    p=np.zeros((len(d),3))
    for ix in groups:
        a=min(.24,2.5/len(ix));p[ix]=[1-2*a,a,a]
    return p

def targets(d,groups,prob,tr):
    ranks=d['evidence_rank'].fill_null(0).to_numpy();target=np.zeros_like(prob);used=np.zeros(len(d),bool);pair_weight=np.zeros(len(d));counts=[];excluded=0
    for ix in groups:
        if not tr[ix[0]]:continue
        e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])]
        pp,ll,w=posterior(prob[ix],e)
        if pp is None:excluded+=1;continue
        target[ix]=pp;used[ix]=True;pair_weight[ix]=1/len(ix);counts.append(len(e))
    row,category=np.nonzero((target>1e-7)&used[:,None]);weight=target[row,category]*pair_weight[row];weight*=len(row)/weight.sum()
    return row,category,weight,used,min(counts),excluded

def checksum(row,category,weight):
    return hashlib.sha256(row.tobytes()+category.tobytes()+weight.tobytes()).hexdigest()

def main():
    ROOT.mkdir(exist_ok=True);d,x,groups=load();start=time.time();audits=[]
    cfg=dict(CONFIG,data_config=json.load(open(data.ROOT/'config.json')),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cp=ROOT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:cp.write_text(json.dumps(cfg,indent=2))
    for held in ['all']+FAMILIES:
        dest=ROOT/held;dest.mkdir(exist_ok=True)
        for fold in range(4):
            tr,va=masks(d,fold,held);prob=initial(d,groups);records=[]
            for step in range(CONFIG['em_steps']):
                rows,category,weight,used,minimum,excluded=targets(d,groups,prob,tr)
                assert not np.any(used&~tr)
                path=dest/f'fold{fold}_em{step+1}.cbm';record=dict(fold=fold,held_family=held,step=step+1,training_hands=int(used.sum()),
                    training_pairs=d.filter(pl.Series(used))['pair_id'].n_unique(),weighted_rows=len(rows),
                    target_sha256=checksum(rows,category,weight),minimum=minimum,incompatible_training_pairs=excluded,
                    training_tables=sorted(set(d['table_id'].to_numpy()[used])),training_pair_ids=sorted(set(d['pair_id'].to_numpy()[used])))
                assert not set(record['training_tables'])&set(d['table_id'].to_numpy()[va])
                if path.exists():
                    model=CatBoostClassifier();model.load_model(str(path));old=json.load(open(path.with_suffix('.json')))
                    assert all(old[k]==v for k,v in record.items())
                else:
                    model=CatBoostClassifier(iterations=CONFIG['iterations'],depth=CONFIG['depth'],learning_rate=CONFIG['learning_rate'],
                        l2_leaf_reg=CONFIG['l2_leaf_reg'],loss_function='MultiClass',random_seed=CONFIG['seed']+fold,
                        thread_count=4,verbose=False,allow_writing_files=False)
                    model.fit(x[rows],category,sample_weight=weight);model.save_model(str(path))
                    record['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();path.with_suffix('.json').write_text(json.dumps(record,indent=2))
                prob=model.predict_proba(x,thread_count=3);assert np.array_equal(model.classes_,np.arange(3));records.append(record)
                print('FAMILY_HOLDOUT',held,fold,step+1,len(rows),round(time.time()-start,1),flush=True)
            parts=[]
            for ix in groups:
                if not va[ix[0]]:continue
                z=d[ix].select('hand_index','pair_id','hand_id','table_id','fold','behavior_family','evidence')
                z=z.with_columns(pl.Series('score',conditioned(prob[ix,1:],minimum)),pl.Series('primary',prob[ix,1]),pl.Series('secondary',prob[ix,2]))
                parts.append(z)
            pl.concat(parts).write_parquet(dest/f'fold{fold}.parquet');audits.extend(records)
        pl.concat([pl.read_parquet(dest/f'fold{f}.parquet') for f in range(4)]).write_parquet(dest/'oof.parquet')
    (ROOT/'audit.json').write_text(json.dumps(audits,indent=2));print('complete',time.time()-start,flush=True)

def evaluate():
    allfam=pl.read_parquet(ROOT/'all/oof.parquet');held=pl.concat([pl.read_parquet(ROOT/fam/'oof.parquet') for fam in FAMILIES])
    q=allfam.rename({'score':'all_families'}).join(held.select('pair_id','hand_id',C('score').alias('withheld_family')),on=['pair_id','hand_id'],validate='1:1')
    r33=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33_known_family_reference'))
    q=q.join(r33,on=['pair_id','hand_id'],validate='1:1');assert len(q)==45129;rows=[];names=['all_families','withheld_family','r33_known_family_reference']
    for (pid,),g in q.group_by('pair_id'):
        truth=set(g.filter(C('evidence')==1)['hand_id']);row=dict(pair_id=pid,table_id=g['table_id'][0],fold=g['fold'][0],family=g['behavior_family'][0])
        for name in names:
            rank=g.sort(name,'hand_id',descending=[True,False])['hand_id'].to_list()[:5];hit=np.array([h in truth for h in rank]);row[name]=float((hit*hit.cumsum()/np.arange(1,6)).sum()/min(5,len(truth)))
        rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'pair_comparison.csv');report={}
    for name in names:
        report[name]=dict(MAP=r[name].mean(),families=dict(r.group_by('family').agg(C(name).mean()).iter_rows()),folds=r.group_by('fold').agg(C(name).mean()).sort('fold')[name].to_list())
    report['limitations']='Conditional retrieval on known true-positive pairs; not unknown-pair detection or the hidden fourth-family score. R33 reference has seen training labels from all families. Generic full-family vs withheld-family is the matched comparison. No R30/risk fallback is used. Prior human feature design knew the three families; removing teacher ancestry does not create an untouched blind test.'
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    import sys
    {'train':main,'evaluate':evaluate}[sys.argv[1]]()
