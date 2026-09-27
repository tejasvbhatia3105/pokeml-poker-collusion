\
\
\
\
\
\
import json,hashlib
from pathlib import Path
import session164_generic_action_em as s
ROOT=Path('artifacts/evidence_session183_depthwise_action_em')
Classifier=s.CatBoostClassifier

def classifier(*args,**kwargs):
    if 'iterations' in kwargs:
        kwargs=dict(kwargs,grow_policy='Depthwise',min_data_in_leaf=1)
    return Classifier(*args,**kwargs)

def setup():
    baseline=Path('artifacts/evidence_session167_action_em_extended')
    v=json.load(open(baseline/'verification.json'));assert v['models']==24 and v['final_prediction_error']==0
    m=Classifier();m.load_model(str(baseline/'fold0_em6.cbm'));p=m.get_all_params()
    assert p['grow_policy']=='SymmetricTree' and p['depth']==5 and p['iterations']==400
    s.ROOT=ROOT;s.CatBoostClassifier=classifier
    s.CONFIG=dict(s.CONFIG,method=__doc__,em_steps=6,grow_policy='Depthwise',min_data_in_leaf=1,
        wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='All24 fixed checkpoints and allfourfolds before comparison; no held-out stopping.')

def verify_policy():
    records=[];base=Path('artifacts/evidence_session167_action_em_extended')
    for fold in range(4):
        for step in range(1,7):
            path=ROOT/f'fold{fold}_em{step}.cbm';m=Classifier();m.load_model(str(path));p=m.get_all_params()
            assert p['grow_policy']=='Depthwise' and p['depth']==5 and p['iterations']==400 and p['min_data_in_leaf']==1
            leaves=m.get_tree_leaf_counts();assert len(leaves)==400 and int(leaves.max())<=32
            if step==1:
                assert json.load(open(path.with_suffix('.json')))['target_sha256']==json.load(open(base/f'fold{fold}_em1.json'))['target_sha256']
            records.append(dict(fold=fold,step=step,grow_policy=p['grow_policy'],depth=p['depth'],iterations=p['iterations'],maximum_leaves=int(leaves.max()),mean_leaves=float(leaves.mean())))
    (ROOT/'growth_policy_verification.json').write_text(json.dumps(dict(records=records,initial_targets_match167=True),indent=2))

if __name__=='__main__':
    import sys
    assert sys.argv[1] in ['train','verify'];setup();s.run(sys.argv[1]=='verify')
    if sys.argv[1]=='verify':verify_policy()
