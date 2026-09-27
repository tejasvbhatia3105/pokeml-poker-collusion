\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
import session197_jev_pilot as p

ROOT = Path('artifacts/evidence_session202_jev_event_history')
FULL = Path('artifacts/evidence_session198_jev_full')
C = pl.col
BASIC = ['relative_time','pot','team_net','net_direction','both_showdown','both_fold','both_survive','seat_distance']


def semantic_features(d):
    d = d.sort('pair_id','time','hand_id')
    raw = ['jev_'+k for k in p.QUESTIONS]
    sym = []
    for k in ['donate','protect','pressure','private']:
        for op,fun in [('max',pl.max_horizontal),('min',pl.min_horizontal)]:
            name = f'jev_{k}_{op}'; sym.append(name)
            d = d.with_columns(fun(C('jev_'+k+'_a'),C('jev_'+k+'_b')).alias(name))
    d = d.with_columns((1-C('jev_ordinary')).alias('jev_not_ordinary'))
    event = ['jev_'+k+'_max' for k in ['donate','protect','pressure','private']]+['jev_not_ordinary']
    history = []
    for c in event:
        expr = [
            (C(c).cum_sum().over('pair_id')-C(c)).alias(c+'_preceding_sum'),
            C(c).cum_max().shift(1).over('pair_id').fill_null(0).alias(c+'_preceding_max'),
            (C(c).rank('average').over('pair_id')/pl.len().over('pair_id')).alias(c+'_pair_percentile')]
        for threshold in [.25,.5]:
            hit = (C(c)>=threshold).cast(pl.Int32)
            expr.append((hit.cum_sum().over('pair_id')-hit).alias(c+'_preceding_ge'+str(threshold)))
        history.extend(e.meta.output_name() for e in expr)
        d = d.with_columns(expr)
    return d,raw+sym+['jev_not_ordinary']+history


def metrics(d,pred):
    rows=[]
    for _,g in d.with_row_index('row').group_by('pair_id',maintain_order=True):
        ix=g['row'].to_numpy();y=g['evidence'].to_numpy();den=min(5,int(y.sum()))
        r={k:g[k][0] for k in ['pair_id','table_id','fold','behavior_family']}
        for name,s in pred.items():
            order=np.lexsort((g['hand_id'].to_numpy(),-s[ix]))[:5];z=y[order]
            r[name]=float((z*np.cumsum(z)/np.arange(1,len(z)+1)).sum()/den)
        rows.append(r)
    pairs=pl.DataFrame(rows);pairs.write_csv(ROOT/'pair_metrics.csv')
    pools=pairs.group_by('table_id').agg(C(list(pred)).sum(),pl.len().alias('n'))
    boot=np.random.default_rng(202).integers(0,len(pools),(5000,len(pools)));report={}
    for name in pred:
        delta=pools[name].to_numpy()-pools['r33'].to_numpy()
        bs=delta[boot].sum(1)/pools['n'].to_numpy()[boot].sum(1)
        report[name]={'MAP':pairs[name].mean(),'delta_vs_r33':pairs[name].mean()-pairs['r33'].mean(),
                      'pool_bootstrap_CI95':np.quantile(bs,[.025,.975]).tolist(),
                      'by_fold':dict(pairs.group_by('fold').agg(C(name).mean()).iter_rows()),
                      'by_family':dict(pairs.group_by('behavior_family').agg(C(name).mean()).iter_rows())}
    return report


def run():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'report.json').exists(),'Do not overwrite a completed experiment'
    raw=pl.read_parquet(FULL/'oof.parquet').select('pair_id','hand_id',*['jev_'+k for k in p.QUESTIONS])
    d=p.canonical().join(raw,on=['pair_id','hand_id'],validate='1:1')
    d=d.join(pl.read_parquet('artifacts/evidence_session195_grounded_selection/grounded.parquet'),on=['pair_id','hand_id'],validate='1:1')
    d,jcols=semantic_features(d)
    gc=json.loads(Path('artifacts/evidence_session195_grounded_selection/columns.json').read_text())
    def matrix(cols):return np.nan_to_num(d.select(cols).to_numpy().astype(np.float32),nan=0,posinf=1e6,neginf=-1e6)
    x=matrix(gc+BASIC);jx=matrix(jcols);compact=matrix(['relative_time']+jcols)
    y=d['evidence'].to_numpy();fold=d['fold'].to_numpy();pred={'r33':d['r33'].to_numpy()}
    for name in ['history_augmented','history_transport','semantic_only']:pred[name]=np.zeros(len(d))
    cfg={'hypothesis':__doc__,'seed':197,'augmented':p.CONFIG['catboost'],
         'compact':dict(p.CONFIG['catboost'],iterations=500,depth=4),'semantic_columns':jcols,
         'base_columns':gc+BASIC,'rows':len(d),'preregistered_variants':3}
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    audits=[]
    for f in range(4):
        tr=fold!=f;va=fold==f
        assert not set(d.filter(pl.Series(tr))['table_id'])&set(d.filter(pl.Series(va))['table_id'])
        prior=d.select('pair_id','hand_id').join(pl.read_parquet(f'artifacts/evidence_session66_current_candidates/design_fold{f}.parquet').select('pair_id','hand_id','prior'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['prior'].to_numpy()
        p0=logit(prior.clip(1e-6,1-1e-6))
        control=CatBoostClassifier();control.load_model(str(FULL/f'control_fold{f}.cbm'))
        correction0=control.predict(x[va],prediction_type='RawFormulaVal',thread_count=2)
        xx=np.column_stack([x,jx]);model=CatBoostClassifier(**cfg['augmented'],random_seed=1970+f)
        model.fit(Pool(xx[tr],y[tr],baseline=p0[tr]))
        correction=model.predict(xx[va],prediction_type='RawFormulaVal',thread_count=2)
        model.save_model(str(ROOT/f'history_fold{f}.cbm'))
        pred['history_augmented'][va]=expit(p0[va]+correction)
        pred['history_transport'][va]=expit(logit(pred['r33'][va].clip(1e-6,1-1e-6))+correction-correction0)
        model=CatBoostClassifier(**cfg['compact'],random_seed=1970+f)
        model.fit(compact[tr],y[tr]);model.save_model(str(ROOT/f'compact_fold{f}.cbm'))
        pred['semantic_only'][va]=model.predict_proba(compact[va],thread_count=2)[:,1]
        audits.append({'fold':f,'pool_overlap':0,'train_rows':int(tr.sum()),'validation_rows':int(va.sum())})
        print('completed fold',f,flush=True)
    d.select('pair_id','hand_id','fold','evidence').with_columns(*[pl.Series(k,v) for k,v in pred.items()]).write_parquet(ROOT/'oof.parquet')
    report=metrics(d,pred);(ROOT/'report.json').write_text(json.dumps(report,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(audits,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':run()
