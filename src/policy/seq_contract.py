from pathlib import Path
import json
import numpy as np

def save_or_check_normalization(out, arrays, columns, aux_evidence=False):
    out=Path(out);path=out/'norm.npz';contract=out/'training_contract.json'
    resumed=any(out.glob('seq_fold*.pt'))
    if resumed:
        if not path.exists():raise RuntimeError('Existing sequence checkpoints have no normalization. Use an intact model archive or a new output directory.')
        with np.load(path) as old:
            bad=[k for k,v in arrays.items() if k not in old or old[k].shape!=np.asarray(v).shape or not np.allclose(old[k],v,rtol=1e-5,atol=1e-6)]
        if bad:raise RuntimeError('Sequence cache/normalization changed for '+','.join(bad)+'. Refusing to overwrite checkpoint normalization; rebuild compatible inputs or train in a new output directory.')
        if contract.exists():
            c=json.loads(contract.read_text())
            if c.get('columns')!=list(columns):raise RuntimeError('Sequence feature order changed on resume.')
            if bool(c.get('aux_evidence'))!=bool(aux_evidence):raise RuntimeError('Auxiliary evidence training mode changed on resume. Use a new output directory.')
            if aux_evidence and c.get('evidence_label_version')!='pair_hand_v2':raise RuntimeError('Auxiliary evidence supervision changed; retrain in a new output directory.')
        elif aux_evidence:raise RuntimeError('Legacy auxiliary-evidence checkpoints lack a pair-hand supervision contract. Retrain in a new output directory.')
                                                                    
        return
    np.savez(path,**arrays)
    contract.write_text(json.dumps({'columns':list(columns),'evidence_label_version':'pair_hand_v2','aux_evidence':bool(aux_evidence)},indent=2))

def load_checked_state(model, state, aux_evidence=False):
    \
    result=model.load_state_dict(state,strict=False)
    allowed=set() if aux_evidence else {'tok.weight','tok.bias'}
    missing=set(result.missing_keys)-allowed
    if missing or result.unexpected_keys:
        raise RuntimeError(f'Incompatible sequence checkpoint: missing={sorted(missing)}, unexpected={result.unexpected_keys}')
    return result
