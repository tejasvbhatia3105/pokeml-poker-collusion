\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json, hashlib
from pathlib import Path
import numpy as np, polars as pl, torch, joblib
from session189_pair_event_prototypes import data
import session11_list_boost as b
from session11_conditional_family import features
from session8_count_conditioning import conditioned

ROOT=Path('artifacts/evidence_session195_grounded_selection');C=pl.col

def build():
    d=data();old=Path('artifacts/evidence_session62_grounded_list_boost')
    model=joblib.load(old/'list_boost_full_fold2.joblib');cols=model['grounded_columns']
    cfg=json.load(open('artifacts/evidence_session37_bet_fold/config.json'))
    ac=cfg['fold_columns'];ec=cfg['paired_columns'];names=ac+ec+['fold_present','pair_current_order','pair_current_gap']
    parts=[];checks=[]
    for fam,action_path,extra_root in [
        ('directed_transfer','artifacts/evidence_session26_exact_fold/fold_actions.parquet','artifacts/evidence_session37_bet_fold'),
        ('soft_play','artifacts/evidence_session38_soft_bet_fold/fold_actions.parquet','artifacts/evidence_session38_soft_bet_fold'),
        ('coordinated_isolation','artifacts/evidence_session41_isolation_bet_fold/fold_actions.parquet','artifacts/evidence_session41_isolation_bet_fold')]:
        a=pl.read_parquet(action_path)
        if 'action_row' not in a.columns:a=a.with_row_index('action_row')
        a=a.sort('action_row');extra=pl.read_parquet(Path(extra_root)/'action_features.parquet').sort('action_row')
        assert a['action_row'].to_list()==extra['action_row'].to_list()==list(range(len(a)))
        align=pl.read_parquet(Path(extra_root)/'alignment.parquet').sort('action_row')
        assert a.select('pair_id','hand_id').equals(align.select('pair_id','hand_id'))
        cur=np.load(f'artifacts/evidence_session50_matchup/{fam}/features.npz')['x'][:,[3,6]]
        x=np.column_stack([a.select(ac).to_numpy(),extra.select(ec).to_numpy(),np.ones(len(a)),cur]).astype(np.float32)
        assert len(a)==len(cur) and a.select('pair_id','hand_id').n_unique()==len(a)
        parts.append(a.select('pair_id','hand_id').with_columns(*[pl.Series('grounded_'+n,x[:,j]) for j,n in enumerate(names)]))
        checks.append(dict(family=fam,fold_actions=len(a),paired_alignment_exact=True))
    q=d.select('pair_id','hand_id').join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left')
    q=q.with_columns(C('grounded_fold_present').fill_null(0)).fill_null(-2)
    pa=pl.read_parquet('artifacts/evidence_session57_isolation_pressure/pressure_actions.parquet')
    px=np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x']
    pc=['precise_mw_'+n for n in ['own','partner','partner_fold_gain','team','call_edge','information_gap','fold_value']]
    assert px.shape==(len(pa),7)
    pa=pa.select('pair_id','hand_id').with_columns(*[pl.Series(c,px[:,i]) for i,c in enumerate(pc)])
    grouped=pa.group_by('pair_id','hand_id').agg(*[getattr(C(c),s)().alias('grounded_'+c+'_'+s) for s in ['mean','max'] for c in pc],pl.len().alias('grounded_pressure_count'))
    q=q.join(grouped,on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(C('grounded_pressure_count').fill_null(0)).fill_null(-2)
    assert set(q.columns)==set(['pair_id','hand_id']+cols)
    x=q.select(cols).to_numpy().astype(np.float32);assert x.shape==(len(d),134) and np.isfinite(x).all()
    q=q.select('pair_id','hand_id',*cols)
    d=d.join(q,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    b.CONFIG['input_root']='artifacts/evidence_session55_current_nested';b.features=features
    expected=pl.read_parquet(old/'conditional_boost_oof.parquet');replays=[]
    for f in [2,3]:
        path=old/f'list_boost_full_fold{f}.joblib';assert not path.stat().st_flags&0x40000000
        model=joblib.load(path);groups,compact,extra,P,M,T,V,D,fv,bid,pos=b.pack(d,f,cols)
        X=np.nan_to_num(np.column_stack([compact,extra]),nan=0,posinf=1e6,neginf=-1e6)
        z=P.numpy().copy();z[bid,pos]+=b.infer(model,X);error=0.
        for i in np.flatnonzero(fv==f):
            g=groups[i];n=len(g)
            prob=torch.softmax(torch.tensor(np.column_stack([np.zeros(n,np.float32),z[i,:n]])),1).numpy()
            inc=conditioned(prob[:,1:],model['minimums'][g['behavior_family'][0]])
            score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
            ref=g.select('pair_id','hand_id').join(expected,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['full'].to_numpy()
            error=max(error,float(abs(score-ref).max()))
        print('GROUNDED_REPLAY',f,error,flush=True);assert error==0
        replays.append(dict(fold=f,prediction_error=error,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    ROOT.mkdir(exist_ok=True);q.write_parquet(ROOT/'grounded.parquet')
    (ROOT/'columns.json').write_text(json.dumps(cols,indent=2))
    (ROOT/'restoration_verification.json').write_text(json.dumps(dict(method=__doc__,actions=checks,replayed_models=replays,
        rows=len(q),features=len(cols),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':build()
