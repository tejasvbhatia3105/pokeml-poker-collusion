\
\
\
\
\
\
import itertools,json,hashlib
from pathlib import Path
import numpy as np
import polars as pl
from session7_likelihood import masks,posterior
ROOT=Path('artifacts/pair_session126_view_supervision');C=pl.col
WINDOWS=[('full',0,3000),('first_2000',0,2000),('last_2000',1000,3000)]

def presence(prob,e,view):
    hyp=masks(len(prob),e)
    if not hyp:return None
    p=np.maximum(np.asarray(prob,float),1e-12);p/=p.sum(1,keepdims=True)
    mass=np.stack([(p*m).sum(1) for _,m in hyp]);logmass=np.log(mass).sum(1);weights=np.exp(logmass-logmass.max());weights/=weights.sum()
    none=[]
    for i,(_,mask) in enumerate(hyp):
        no=p[view,0]*mask[view,0]/mass[i,view];none.append(float(np.prod(no)))
    return float(np.clip(1-np.dot(weights,none),0,1))

def check():
    rng=np.random.default_rng(12601);p=rng.dirichlet([8,1,1],size=6);lists={}
    for cats in itertools.product(range(3),repeat=6):
        order=tuple(([i for i,c in enumerate(cats) if c==1]+[i for i,c in enumerate(cats) if c==2])[:5])
        if not order:continue
        mass=float(np.prod(p[np.arange(6),cats]));lists.setdefault(order,[]).append((cats,mass))
    error=0.;checks=0
    for e,paths in lists.items():
        for view in [np.array([0,2,4]),np.array([4,5]),np.arange(6)]:
            actual=presence(p,e,view);expected=sum(m for c,m in paths if any(c[i] for i in view))/sum(m for c,m in paths)
            error=max(error,abs(actual-expected));checks+=1
    assert error<1e-12;return dict(enumerated_paths=729,conditional_view_checks=checks,maximum_probability_error=error)

def load():
    h=pl.read_parquet('artifacts/evidence_session115_relationship_data/hands.parquet')
    b=pl.read_parquet('artifacts/evidence_session115_relationship_data/bags.parquet').filter(C('label')==1)
    h=h.join(b.select('pair_id','fold','behavior_family'),on='pair_id',how='inner')
    return h.join(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id','evidence_rank'),on=['pair_id','hand_id'],how='left',validate='1:1')

def main():
    ROOT.mkdir(exist_ok=True);proof=check();h=load();records=[];cert=json.load(open('artifacts/evidence_session55_current_nested/input_verification.json'))
    assert cert['saved_models_replayed']==175 and cert['event_probability_error']==0 and cert['all_training_prediction_and_outer_pool_sets_disjoint']
    for outer in range(4):
        raw=pl.read_parquet(f'artifacts/evidence_session55_current_nested/nested_outer{outer}.parquet')
        d=h.filter(C('fold')!=outer).join(raw.select('pair_id','hand_id','cat_primary','cat_secondary','hist_primary','hist_secondary'),on=['pair_id','hand_id'],validate='1:1')
        for (pid,),g in d.group_by('pair_id'):
            g=g.sort('time_index','hand_id');e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])]
            cp=g.select('cat_primary','cat_secondary').to_numpy();cp=cp/np.maximum(1,cp.sum(1))[:,None]
            hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None]
            probabilities={'cat':np.column_stack([1-cp.sum(1),cp]),'joint':np.column_stack([1-((cp+hp)/2).sum(1),(cp+hp)/2])}
            times=g['time_index'].to_numpy()
            for name,a,b in WINDOWS:
                rawview=np.flatnonzero((times>=a)&(times<b))
                if len(rawview)<15:continue
                view=rawview[-160:];listed=int(np.isin(view,e).sum());row=dict(outer=outer,pair_id=pid,native_fold=int(g['fold'][0]),family=g['behavior_family'][0],
                    window=name,n_view=len(rawview),n_input=len(view),listed_before_crop=int(np.isin(rawview,e).sum()),listed_in_input=listed,
                    compatible_list=bool(masks(len(g),e)))
                for label,p in probabilities.items():
                    value=presence(p,e,view);row[label+'_presence']=value
                    if listed and value is not None:assert value==1
                records.append(row)
    r=pl.DataFrame(records);assert not len(r.filter(C('outer')==C('native_fold')));r.write_parquet(ROOT/'posterior_training_views.parquet')
                                                                             
                                                                           
    v=r.group_by('pair_id','window').agg(C('family').first(),C('n_view').first(),C('n_input').first(),C('listed_before_crop').first(),
        C('listed_in_input').first(),C('compatible_list').first(),C('cat_presence').mean(),C('joint_presence').mean(),
        (C('joint_presence').max()-C('joint_presence').min()).alias('teacher_range'),pl.len().alias('references'))
    assert (v['references']==3).all();v.write_parquet(ROOT/'view_summary.parquet');summaries=[]
    for name,_,_ in WINDOWS:
        z=v.filter(C('window')==name);unc=z.filter((C('listed_in_input')==0)&C('compatible_list'))
        summaries.append(dict(window=name,views=len(z),no_listed_before_crop=int((z['listed_before_crop']==0).sum()),no_listed_in_input=int((z['listed_in_input']==0).sum()),
            crop_removed_all_listed=int(((z['listed_before_crop']>0)&(z['listed_in_input']==0)).sum()),compatible_uncertain=len(unc),
            uncertain_joint_presence_quantiles=np.quantile(unc['joint_presence'].to_numpy(),[0,.25,.5,.75,1]).tolist() if len(unc) else [],
            uncertain_below01=int((unc['joint_presence']<.1).sum()),uncertain_below05=int((unc['joint_presence']<.5).sum())))
    report=dict(method=__doc__,windows=summaries,verification=proof,all_training_views_exclude_outer_labels=True,
        controls='Cat-only and equal Cat/HGB prior; no parameter fitting or selection',
        limitation='Annotation-conditioned event presence depends on the two-tier model and teacher calibration. Incompatible lists are flagged and not assigned invented posterior values. No new pair model, candidate, or leaderboard gain.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
