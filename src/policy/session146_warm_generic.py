\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,hashlib
from pathlib import Path
import numpy as np,polars as pl
import session123_family_holdout as s

ROOT=Path('artifacts/evidence_session146_warm_generic');SOURCE=Path('artifacts/evidence_session55_current_nested');C=pl.col
BASE_MASKS=s.masks;BASE_INITIAL=s.initial;BASE_CONFIG=dict(s.CONFIG)
CONTEXT={};TEACHERS=None

def masks(d,fold,held):
    tr,va=BASE_MASKS(d,fold,held);CONTEXT.update(fold=fold,held=held,tr=tr);return tr,va

def initial(d,groups):
    global TEACHERS
    if TEACHERS is None:TEACHERS=np.load(ROOT/'teacher_probabilities.npy')
    p=BASE_INITIAL(d,groups);tr=CONTEXT['tr'];p[tr]=TEACHERS[CONTEXT['fold'],tr]
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1);return p

def setup():
    assert (ROOT/'preparation_verification.json').exists()
    s.ROOT=ROOT;s.masks=masks;s.initial=initial;s.CONFIG=dict(BASE_CONFIG)
    s.CONFIG.update(method=__doc__,initialization='Only available-family outer-training rows: normalized joint current nested55 Cat/HGB probabilities; protected rows retain constant cold prior.',
        teacher_root=str(SOURCE),teacher_manifest_sha256=hashlib.sha256((ROOT/'teacher_manifest.json').read_bytes()).hexdigest(),
        wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def prepare():
    ROOT.mkdir(exist_ok=True);d,x,groups=s.load();proof=json.load(open(SOURCE/'input_verification.json'))
    assert proof['saved_models_replayed']==175 and proof['event_probability_error']==0 and proof['all_training_prediction_and_outer_pool_sets_disjoint']
    parts=[];manifest={}
    for f in range(4):
        path=SOURCE/f'nested_outer{f}.parquet';q=d.select('pair_id','hand_id','fold').join(pl.read_parquet(path).select('pair_id','hand_id',C('fold').alias('source_fold'),'cat_primary','cat_secondary','hist_primary','hist_secondary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
        assert len(q)==len(d) and (q['fold']==q['source_fold']).all()
        ca=q.select('cat_primary','cat_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=q.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];jp=.5*(ca+hp)
        p=np.maximum(np.column_stack([1-jp.sum(1),jp]),1e-8);p/=p.sum(1)[:,None];parts.append(p);manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    a=np.stack(parts);assert a.shape==(4,45129,3);np.save(ROOT/'teacher_probabilities.npy',a)
    manifest['teacher_array_sha256']=hashlib.sha256((ROOT/'teacher_probabilities.npy').read_bytes()).hexdigest()
                                                                              
    references=0
    folds=json.load(open('artifacts/policy/table_folds.json'))
    for outer in range(4):
        for parent in range(4):
            if parent==outer:continue
            for half in range(2):
                path=SOURCE/f'outer{outer}_parent{parent}_half{half}.json';record=json.load(open(path));manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                for v in record['records']:
                    assert set(v['training_tables']).isdisjoint(v['prediction_tables'])
                    assert all(folds[t]!=outer for t in v['training_tables']);references+=1
    (ROOT/'teacher_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    global TEACHERS
    TEACHERS=a;checks=[]
    for held in ['all']+s.FAMILIES:
        for f in range(4):
            tr,va=masks(d,f,held);p=initial(d,groups);cold=BASE_INITIAL(d,groups);np.testing.assert_array_equal(p[~tr],cold[~tr]);saved=a[f,~tr].copy();a[f,~tr]=[.1,.3,.6];mutated=initial(d,groups);a[f,~tr]=saved;np.testing.assert_array_equal(p,mutated)
            dm=d.with_columns(pl.when(pl.Series(~tr)).then(pl.lit(999)).otherwise(C('evidence_rank')).alias('evidence_rank'))
            one=s.targets(d,groups,p,tr);two=s.targets(dm,groups,p,tr)
            for u,v in zip(one[:4],two[:4]):np.testing.assert_array_equal(u,v)
            assert one[4:]==two[4:]
            if held!='all':assert not d[one[0]].filter(C('behavior_family')==held).height
            checks.append(dict(held=held,fold=f,protected_teacher_mutation_exact=True,protected_rank_mutation_exact=True,protected_prior_matches_cold=True,training_rows=int(tr.sum()),weighted_rows=len(one[0])))
    report=dict(shape=list(a.shape),source_family_pool_exclusion_records=references,checks=checks,method=__doc__,input_source='same raw122 gameplay array as cold123; no teacher probabilities appended as features')
    (ROOT/'preparation_verification.json').write_text(json.dumps(report,indent=2));(ROOT/'PROTOCOL.md').write_text('''# Available-specialist initialization for a generic retriever

Compare with original123 using the same family-blind raw122 features, folds, seed, two EM steps,400 depth5 CatBoost trees and all other fit settings. Only initial class probabilities on allowed training rows change. Use normalized, floored1e-8 joint Cat/HGB probabilities from strictly nested55 current experts. The source family of each training row is available; the withheld family and outer validation rows retain cold constants and never generate a target. This is an initialization intervention, not an added inference feature or blended submission.

Fit all four outer folds for the all-family control and each of three fully withheld families:32 checkpoints total. No stopping or schedule change based on partial validation. Retain every checkpoint; verify targets, pool/family exclusions, hashes and final prediction replay with the adapted original verifier. Report against matched cold control and current retained-expert transfer baseline. A benefit on unknown-family simulation still needs a validated routing rule and evaluation inference before any candidate.

Existing55 experts use the source family's own supervised targets and exclude the outer fold plus the predicted training pool half. Their old histogram teachers retain the verified nested10 ancestry. Human feature/target design knew the public families, so neither this nor the cold comparator is an untouched blind benchmark.
''');print('WARM_PREPARED',a.shape,references,len(checks),flush=True)

if __name__=='__main__':
    import sys
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    else:
        setup()
        if mode=='train':s.main()
        elif mode=='evaluate':s.evaluate()
        elif mode=='verify':
            import session123_verify as v
            v.main()
        else:raise ValueError(mode)
