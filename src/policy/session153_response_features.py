\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib,time
from pathlib import Path
import numpy as np,polars as pl
import session119_private_partner_data as raw
import session122_family_blind_hands as old

ROOT=Path('artifacts/evidence_session153_response_features');C=pl.col
FIELDS=['log_amount_bb','amount_pot_fraction','amount_stack_fraction','raise_increment_pot_fraction',
        'call_coverage','size_own_equity','size_partner_equity','size_partner_minus_own_equity']

def responses(amount,x):
    cols=raw.PUBLIC+raw.PRIVATE
    def a(name):return x[:,cols.index(name)].astype(np.float64)
    bb=a('big_blind');assert (bb>0).all()
    amount_bb=amount/bb;size=np.log1p(amount_bb)
    return np.column_stack([size,amount_bb/(a('pot_bb')+1),amount_bb/(a('stack_bb')+1),
        np.maximum(0,amount_bb-a('call_bb'))/(a('pot_bb')+1),
        np.minimum(amount_bb,a('call_bb'))/(a('call_bb')+1),size*a('equity'),
        size*a('partner_private_equity'),size*a('partner_equity_minus_own')]).astype(np.float32)

def build():
    ROOT.mkdir(exist_ok=True);d=pl.read_parquet(old.ROOT/'hands.parquet');arrays=[];metas=[];audit=[];start=time.time()
    for table in sorted(d['table_id'].unique()):
        meta=pl.read_parquet(raw.ROOT/f'{table}.parquet').with_row_index('local')
        q=meta.join(d.select('pair_id','hand_id','hand_index'),on=['pair_id','hand_id'],how='inner',validate='m:1')
        x=np.load(raw.ROOT/f'{table}.npz')['x'][q['local'].to_numpy()]
        policy=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet',columns=['hand_id','action_no','amount','big_blind','log_amount_bb'])
        policy=policy.with_columns(C('action_no').cast(q.schema['action_no']))
        z=q.join(policy,on=['hand_id','action_no'],validate='m:1',maintain_order='left')
        assert len(z)==len(q) and z['amount'].null_count()==0
        extra=responses(z['amount'].to_numpy(),x)
        err=float(np.max(np.abs(extra[:,0]-z['log_amount_bb'].to_numpy())))
        assert err<1e-5,(table,err)
        arrays.append(extra);metas.append(q.select('hand_index','action_class').with_columns(pl.Series('post',x[:,raw.PUBLIC.index('street_no')]>0)))
        audit.append(dict(table=table,actions=len(q),log_amount_replay_error=err))
    x=np.concatenate(arrays);meta=pl.concat(metas);bag=meta['hand_index'].to_numpy();cls=meta['action_class'].to_numpy();post=meta['post'].to_numpy()
    masks=[np.ones(len(x),bool),cls==0,(cls==0)&post,cls==2,(cls==2)&post,cls==3,(cls==3)&post]
    pieces=[];columns=[];checks=0
    for name,mask in zip(old.GROUPS,masks):
        block,_=old.aggregate(x,bag,mask,len(d));pieces.append(block[:,1:])
        columns += [name+'_'+stat+'_response_'+c for stat in ['mean','min','max'] for c in FIELDS]
        for i in np.random.default_rng(15301).choice(len(d),32,replace=False):
            z=x[mask&(bag==i)]
            expected=np.r_[z.mean(0,dtype=np.float64),z.min(0),z.max(0)] if len(z) else np.full(3*x.shape[1],-2)
            np.testing.assert_allclose(block[i,1:],expected,rtol=1e-6,atol=1e-5);checks+=1
    out=np.column_stack(pieces);assert out.shape==(45129,168) and np.isfinite(out).all()
    np.save(ROOT/'extra.npy',out)
    config=dict(method=__doc__,columns=columns,rows=len(d),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_config_sha256=hashlib.sha256((old.ROOT/'config.json').read_bytes()).hexdigest(),extra_sha256=hashlib.sha256((ROOT/'extra.npy').read_bytes()).hexdigest())
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));(ROOT/'verification.json').write_text(json.dumps(dict(tables=audit,scalar_aggregate_checks=checks,actions=len(x),features=out.shape[1]),indent=2))
    print('RESPONSE_FEATURES_COMPLETE',out.shape,len(x),round(time.time()-start,1),flush=True)

if __name__=='__main__':build()
