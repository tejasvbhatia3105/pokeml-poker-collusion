\
\
\
\
\
import os,json,hashlib,shutil
from pathlib import Path
import session164_generic_action_em as s
ROOT=Path('artifacts/evidence_session167_action_em_extended');BASE=s.ROOT

def setup():
    v=json.load(open(BASE/'verification.json'));assert v['models']==8 and v['final_prediction_error']==0
    ROOT.mkdir(exist_ok=True)
    for fold in range(4):
        for step in [1,2]:
            old=BASE/f'fold{fold}_em{step}.cbm';new=ROOT/old.name
            if not new.exists():os.link(old,new)
            assert hashlib.sha256(old.read_bytes()).digest()==hashlib.sha256(new.read_bytes()).digest()
            metadata=new.with_suffix('.json')
            if not metadata.exists():shutil.copyfile(old.with_suffix('.json'),metadata)
    s.ROOT=ROOT;s.CONFIG=dict(s.CONFIG)
    s.CONFIG.update(method=__doc__,em_steps=6,selection='Fixed six-step extension; all four folds complete before comparison; no validation stopping.',
        base_checkpoint_root=str(BASE),wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    import sys
    assert sys.argv[1] in ['train','verify'];setup();s.run(sys.argv[1]=='verify')
