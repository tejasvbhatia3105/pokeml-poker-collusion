\
\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,time
from pathlib import Path
import numpy as np,polars as pl
from scipy.special import softmax
from catboost import CatBoostClassifier,Pool
import session207_jev_family_transfer as source
from session197_jev_pilot import QUESTIONS,available

ROOT=Path('artifacts/evidence_session208_jev_frozen_transfer');C=pl.col
NAMES=['base','control','jev','contrast']
CONFIG={'method':__doc__,'iterations':120,'depth':3,'learning_rate':.035,'l2_leaf_reg':20,'seed':20800,'threads':2}


def metrics(q):
    rows=[]
    for _,g in q.group_by('pair_id'):
        r={k:g[k][0] for k in ['pair_id','table_id','fold','behavior_family']};y=g['evidence'].to_numpy();den=min(5,int(y.sum()))
        for name in NAMES:
            order=np.lexsort((g['hand_id'].to_numpy(),-g[name].to_numpy()))[:5];z=y[order]
            r[name]=float((z*np.cumsum(z)/np.arange(1,len(z)+1)).sum()/den)
        rows.append(r)
    pairs=pl.DataFrame(rows);pairs.write_csv(ROOT/'pair_metrics.csv');report={}
    for family,g in [('all_withheld',pairs),*[(k[0],v) for k,v in pairs.group_by('behavior_family')]]:
        pool=g.group_by('table_id').agg(C(NAMES).sum(),pl.len().alias('n'))
        ix=np.random.default_rng(208).integers(0,len(pool),(5000,len(pool)));r={n:g[n].mean() for n in NAMES}
        for a,b in [('jev','control'),('contrast','base'),('jev','base')]:
            diff=pool[a].to_numpy()-pool[b].to_numpy();bs=diff[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
            r[a+'_vs_'+b]={'gain':g[a].mean()-g[b].mean(),'pool_CI95':np.quantile(bs,[.025,.975]).tolist()}
        r['folds']=g.group_by('fold').agg(C(NAMES).mean()).sort('fold').to_dicts();report[family]=r
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def run():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'report.json').exists()
    (ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d,x,groups=source.load()
    j=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session198_jev_full/oof.parquet').select('pair_id','hand_id',*['jev_'+k for k in QUESTIONS]),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    jx=j.select(['jev_'+k for k in QUESTIONS]).to_numpy().astype(np.float32)
    chronology=np.zeros((len(d),3),np.float32)
    for ix in groups:chronology[ix]=np.column_stack([d[ix]['time_index'].to_numpy()/3000,np.arange(len(ix))/max(1,len(ix)-1),np.full(len(ix),np.log1p(len(ix)))])
    out=[];audits=[];start=time.monotonic()
    for family in source.FAMILIES:
        for fold in range(4):
            tr=(d['fold'].to_numpy()!=fold)&(d['behavior_family'].to_numpy()!=family)
            va=(d['fold'].to_numpy()==fold)&(d['behavior_family'].to_numpy()==family)
            model=CatBoostClassifier();model.load_model(str(available(source.BASE/family/f'fold{fold}_em2.cbm')))
            prob=model.predict_proba(x,thread_count=2);p0=np.log(prob.clip(1e-10,1))
            expected=pl.read_parquet(available(source.BASE/family/f'fold{fold}.parquet'))
            replay=d.filter(pl.Series(va)).select('pair_id','hand_id').join(expected,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
            np.testing.assert_allclose(prob[va,1:],replay.select('primary','secondary').to_numpy(),atol=1e-10,rtol=0)
            rows,category,weight,used,minimum,excluded=source.targets(d,groups,prob,tr)
            assert not np.any(used&~tr)
            assert family not in set(d.filter(pl.Series(used))['behavior_family'])
            assert not set(d.filter(pl.Series(used))['table_id'])&set(d.filter(pl.Series(va))['table_id'])
            before=np.zeros((len(d),2),np.float32)
            for ix in groups:before[ix]=(np.cumsum(prob[ix,1:],axis=0)-prob[ix,1:])/5
            control=np.column_stack([p0,chronology,before]).astype(np.float32)
            raw={}
            for name,X in [('control',control),('jev',np.column_stack([control,jx]))]:
                m=CatBoostClassifier(iterations=CONFIG['iterations'],depth=CONFIG['depth'],learning_rate=CONFIG['learning_rate'],
                    l2_leaf_reg=CONFIG['l2_leaf_reg'],loss_function='MultiClass',random_seed=CONFIG['seed']+fold,
                    thread_count=2,verbose=False,allow_writing_files=False)
                m.fit(Pool(X[rows],category,weight=weight,baseline=p0[rows]))
                raw[name]=m.predict(X[va],prediction_type='RawFormulaVal',thread_count=2)
                path=ROOT/f'{family}_fold{fold}_{name}.cbm';m.save_model(str(path));again=CatBoostClassifier();again.load_model(str(path))
                np.testing.assert_array_equal(raw[name],again.predict(X[va],prediction_type='RawFormulaVal',thread_count=2))
            prob_variants={'base':prob}
            for name,delta in [('control',raw['control']),('jev',raw['jev']),('contrast',raw['jev']-raw['control'])]:
                z=prob.copy();z[va]=softmax(p0[va]+delta,axis=1);prob_variants[name]=z
            for ix in groups:
                if not va[ix[0]]:continue
                g=d[ix].select('pair_id','hand_id','table_id','fold','behavior_family','evidence')
                g=g.with_columns(*[pl.Series(name,source.conditioned(values[ix,1:],minimum)) for name,values in prob_variants.items()])
                expected_score=g.select('pair_id','hand_id').join(expected.select('pair_id','hand_id','score'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['score'].to_numpy()
                np.testing.assert_allclose(g['base'].to_numpy(),expected_score,atol=1e-10,rtol=0);out.append(g)
            audits.append({'family':family,'fold':fold,'teacher_predictions_replayed':True,'pool_overlap':0,
                           'held_family_in_training':False,'new_checkpoints_replayed':True,
                           'training_target_sha256':source.checksum(rows,category,weight),'minimum':minimum})
            (ROOT/'audit.json').write_text(json.dumps(audits,indent=2));print(family,fold,'seconds',round(time.monotonic()-start,1),flush=True)
    q=pl.concat(out);assert len(q)==45129;q.write_parquet(ROOT/'oof.parquet');metrics(q)


if __name__=='__main__':run()
