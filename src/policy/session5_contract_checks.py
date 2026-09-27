import tempfile,json,hashlib
from pathlib import Path
import numpy as np, torch
from torch import nn
from seq_contract import save_or_check_normalization,load_checked_state
report={}
def rejects(fn):
    try:fn()
    except RuntimeError:return True
    return False
with tempfile.TemporaryDirectory() as t:
    p=Path(t);a={'mu':np.zeros(2),'sd':np.ones(2)}
    save_or_check_normalization(p,a,['a','b']);(p/'seq_fold0.pt').touch()
    before=hashlib.sha256((p/'norm.npz').read_bytes()).hexdigest()
    save_or_check_normalization(p,a,['a','b']);report['same_inputs_resume_passed']=True
    report['changed_normalization_rejected']=rejects(lambda:save_or_check_normalization(p,{'mu':np.ones(2),'sd':np.ones(2)},['a','b']))
    report['feature_order_rejected']=rejects(lambda:save_or_check_normalization(p,a,['b','a']))
    report['aux_mode_change_rejected']=rejects(lambda:save_or_check_normalization(p,a,['a','b'],True))
    report['saved_normalization_preserved']=hashlib.sha256((p/'norm.npz').read_bytes()).hexdigest()==before
    (p/'training_contract.json').unlink()
    report['legacy_aux_supervision_rejected']=rejects(lambda:save_or_check_normalization(p,a,['a','b'],True))
m=nn.Module();m.inp=nn.Linear(2,2);m.tok=nn.Linear(2,1);state=m.state_dict()
load_checked_state(m,state,True);report['complete_checkpoint_passed']=True
legacy={k:v for k,v in state.items() if not k.startswith('tok.')}
load_checked_state(m,legacy);report['unused_legacy_head_allowed']=True
report['missing_trained_head_rejected']=rejects(lambda:load_checked_state(m,legacy,True))
report['missing_encoder_rejected']=rejects(lambda:load_checked_state(m,{k:v for k,v in state.items() if k!='inp.weight'}))
report['unexpected_keys_rejected']=rejects(lambda:load_checked_state(m,{**state,'obsolete':torch.zeros(1)}))
assert all(report.values()),report
Path('artifacts/evidence_session5/resume_contract_checks.json').write_text(json.dumps(report,indent=2));print(report)
