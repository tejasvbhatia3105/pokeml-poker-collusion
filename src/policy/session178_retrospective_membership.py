\
\
\
\
import json,hashlib
from pathlib import Path
import numpy as np
import session174_direct_list_membership as s
import session177_retrospective_features as extra
ROOT=Path('artifacts/evidence_session178_retrospective_membership')
old_load=s.s.load

def load():
    d,x,groups=old_load();z=np.load(extra.ROOT/'extra.npy',mmap_mode='r');assert z.shape==(len(d),42)
    return d,np.column_stack([x,z]),groups

def setup():
    cfg=json.load(open(extra.ROOT/'config.json'));proof=json.load(open(extra.ROOT/'verification.json'))
    assert len(proof['checks'])==4 and proof['rows']==45129 and proof['no_fitted_teachers']
    assert cfg['extra_sha256']==hashlib.sha256((extra.ROOT/'extra.npy').read_bytes()).hexdigest()
    assert cfg['base_config_sha256']==hashlib.sha256((s.s.data.ROOT/'config.json').read_bytes()).hexdigest()
    s.s.load=load;s.ROOT=ROOT;s.CONFIG=dict(s.CONFIG,method=__doc__,extra_config=cfg,wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':
    import sys
    assert sys.argv[1] in ['train','verify'];setup();s.main(sys.argv[1]=='verify')
