\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,time,resource,hashlib
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session7_likelihood import posterior
from session8_count_conditioning import conditioned
from session197_jev_pilot import QUESTIONS,available

ROOT=Path('artifacts/evidence_session207_jev_family_transfer');C=pl.col
BASE=Path('artifacts/evidence_session123_family_holdout')
DATA=Path('artifacts/evidence_session122_family_blind_hands')
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']

                                                                         
                                                                           
def load():
    d=pl.read_parquet(available(DATA/'hands.parquet'));x=np.load(available(DATA/'x.npy'),mmap_mode='r')
    assert np.array_equal(d['hand_index'],np.arange(len(d)))
    groups=[g.sort('time_index','hand_id')['hand_index'].to_numpy() for _,g in d.group_by('pair_id')]
    groups.sort(key=lambda g:d['pair_id'][int(g[0])]);return d,x,groups

def initial(d,groups):
    prob=np.zeros((len(d),3))
    for ix in groups:
        a=min(.24,2.5/len(ix));prob[ix]=[1-2*a,a,a]
    return prob

def targets(d,groups,prob,tr):
    ranks=d['evidence_rank'].fill_null(0).to_numpy();target=np.zeros_like(prob);used=np.zeros(len(d),bool);pw=np.zeros(len(d));counts=[];excluded=0
    for ix in groups:
        if not tr[ix[0]]:continue
        e=np.flatnonzero(ranks[ix]>0);e=e[np.argsort(ranks[ix][e])]
        pp,ll,w=posterior(prob[ix],e)
        if pp is None:excluded+=1;continue
        target[ix]=pp;used[ix]=True;pw[ix]=1/len(ix);counts.append(len(e))
    row,category=np.nonzero((target>1e-7)&used[:,None]);weight=target[row,category]*pw[row];weight*=len(row)/weight.sum()
    return row,category,weight,used,min(counts),excluded

def checksum(row,category,weight):
    return hashlib.sha256(row.tobytes()+category.tobytes()+weight.tobytes()).hexdigest()


def run():
    ROOT.mkdir(exist_ok=True);d,x,groups=load()
    j=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session198_jev_full/oof.parquet').select('pair_id','hand_id',*['jev_'+k for k in QUESTIONS]),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    jx=j.select(['jev_'+k for k in QUESTIONS]).to_numpy().astype(np.float32)
    X=np.column_stack([x,jx]);assert np.isfinite(X).all()
    cfg=dict(json.loads(available(BASE/'config.json').read_text()),method=__doc__,thread_count=2,base_features=x.shape[1],jev_features=list(QUESTIONS),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2));start=time.monotonic();out=[]
    for held in FAMILIES:
        dest=ROOT/held;dest.mkdir(exist_ok=True)
        for fold in range(4):
            tr=(d['fold'].to_numpy()!=fold)&(d['behavior_family'].to_numpy()!=held)
            va=(d['fold'].to_numpy()==fold)&(d['behavior_family'].to_numpy()==held)
            prob=initial(d,groups)
            assert held not in set(d.filter(pl.Series(tr))['behavior_family'])
            for step in range(cfg['em_steps']):
                rows,category,weight,used,minimum,excluded=targets(d,groups,prob,tr)
                assert not np.any(used&~tr)
                assert not set(d.filter(pl.Series(used))['table_id'])&set(d.filter(pl.Series(va))['table_id'])
                record={'fold':fold,'held_family':held,'step':step+1,'target_sha256':checksum(rows,category,weight),
                        'minimum':minimum,'excluded_incompatible_pairs':excluded,'train_hands':int(used.sum()),
                        'pool_overlap':0,'held_family_in_training':False}
                path=dest/f'fold{fold}_em{step+1}.cbm'
                if path.exists():
                    old=json.loads(path.with_suffix('.json').read_text());assert all(old[k]==v for k,v in record.items())
                    model=CatBoostClassifier();model.load_model(str(path))
                else:
                    model=CatBoostClassifier(iterations=cfg['iterations'],depth=cfg['depth'],learning_rate=cfg['learning_rate'],
                        l2_leaf_reg=cfg['l2_leaf_reg'],loss_function='MultiClass',random_seed=cfg['seed']+fold,
                        thread_count=2,used_ram_limit='2gb',verbose=False,allow_writing_files=False)
                    model.fit(X[rows],category,sample_weight=weight);model.save_model(str(path))
                    record['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                    path.with_suffix('.json').write_text(json.dumps(record,indent=2))
                prob=model.predict_proba(X,thread_count=2);assert np.array_equal(model.classes_,np.arange(3))
                replay=CatBoostClassifier();replay.load_model(str(path))
                np.testing.assert_array_equal(prob[va],replay.predict_proba(X[va],thread_count=2))
                print(held,'fold',fold,'step',step+1,'seconds',round(time.monotonic()-start,1),
                      'peak_rss_mb',round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6),flush=True)
            parts=[]
            for ix in groups:
                if not va[ix[0]]:continue
                parts.append(d[ix].select('pair_id','hand_id','table_id','fold','behavior_family','evidence')
                             .with_columns(pl.Series('jev',conditioned(prob[ix,1:],minimum))))
            q=pl.concat(parts).join(pl.read_parquet(available(BASE/held/f'fold{fold}.parquet')).select('pair_id','hand_id',C('score').alias('control')),on=['pair_id','hand_id'],validate='1:1')
            q.write_parquet(dest/f'fold{fold}.parquet');out.append(q)
    q=pl.concat(out);assert len(q)==45129;q.write_parquet(ROOT/'oof.parquet');metrics(q)


def metrics(q):
    rows=[]
    for _,g in q.group_by('pair_id'):
        r={k:g[k][0] for k in ['pair_id','table_id','fold','behavior_family']}
        y=g['evidence'].to_numpy();den=min(5,int(y.sum()))
        for name in ['control','jev']:
            order=np.lexsort((g['hand_id'].to_numpy(),-g[name].to_numpy()))[:5];z=y[order]
            r[name]=float((z*np.cumsum(z)/np.arange(1,len(z)+1)).sum()/den)
        rows.append(r)
    pairs=pl.DataFrame(rows);pairs.write_csv(ROOT/'pair_metrics.csv');report={}
    for fam,g in [('all_withheld',pairs),*[(k[0],v) for k,v in pairs.group_by('behavior_family')]]:
        pool=g.group_by('table_id').agg(C('control','jev').sum(),pl.len().alias('n'))
        b=np.random.default_rng(207).integers(0,len(pool),(5000,len(pool)))
        delta=pool['jev'].to_numpy()-pool['control'].to_numpy();bs=delta[b].sum(1)/pool['n'].to_numpy()[b].sum(1)
        report[fam]={'control_MAP':g['control'].mean(),'jev_MAP':g['jev'].mean(),'gain':g['jev'].mean()-g['control'].mean(),
                     'pool_bootstrap_CI95':np.quantile(bs,[.025,.975]).tolist(),
                     'folds':g.group_by('fold').agg(C('control','jev').mean()).sort('fold').to_dicts()}
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
