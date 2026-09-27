\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib
from pathlib import Path
import numpy as np
import session123_family_holdout as s
import session153_response_features as features

ROOT=Path('artifacts/evidence_session154_response_control')
BASE_LOAD=s.load;BASE_CONFIG=dict(s.CONFIG)

def load():
    d,x,groups=BASE_LOAD();extra=np.load(features.ROOT/'extra.npy',mmap_mode='r')
    assert extra.shape==(len(d),168)
    return d,np.column_stack([x,extra]),groups

def setup():
    cfg=json.load(open(features.ROOT/'config.json'))
    assert cfg['extra_sha256']==hashlib.sha256((features.ROOT/'extra.npy').read_bytes()).hexdigest()
    assert cfg['source_sha256']==hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest()
    s.ROOT=ROOT;s.load=load;s.FAMILIES=[];s.CONFIG=dict(BASE_CONFIG)
    s.CONFIG.update(method=__doc__,response_feature_config=cfg,
        wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='Fixed all-family control: four folds, two EM steps; no validation stopping. No withheld-family claim.')

if __name__=='__main__':
    import sys
    mode=sys.argv[1];setup()
    if mode=='train':s.main()
    elif mode=='verify':
        import session123_verify as v
        sys.argv=['session123_verify','control-only'];v.main()
    else:raise ValueError(mode)
