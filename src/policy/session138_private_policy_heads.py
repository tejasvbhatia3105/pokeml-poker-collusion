\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,hashlib,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,polars as pl
import session105_rollout_heads as base
import session121_private_witness_values as values

ROOT=Path('artifacts/evidence_session138_private_policy_heads')
ARMS=['public','public_private']
FIELDS=[a+'_'+f for a in ['public','private'] for f in values.FIELDS]

def extra(a,kind,f,arm):
    q=pl.read_parquet(ROOT/'queries.parquet').filter(pl.col('kind')==kind)
    z=a.select('pair_id','hand_id',pl.col('action_no').cast(pl.Int64)).join(q.select('pair_id','hand_id','action_no','query_id'),on=['pair_id','hand_id','action_no'],maintain_order='left',validate='1:1')
    assert z['query_id'].null_count()==0 and len(z)==len(a)
    x=np.load(ROOT/f'features_fold{f}.npz')['x'][z['query_id'].to_numpy()]
    return x[:,:6] if arm=='public' else x

def setup():
    base.ROOT=ROOT;base.KINDS=ARMS;base.extra=extra
                                                                               
                                                                               
    base.roll=SimpleNamespace(ROOT=values.ROOT,FIELDS=values.FIELDS,provenance=values.provenance)

def prepare():
    ROOT.mkdir(exist_ok=True);q=pl.read_parquet(values.ROOT/'queries.parquet')
    cfg=json.loads((values.ROOT/'config.json').read_text());assert cfg['provenance_sha256']==values.provenance()
    assert q['query_id'].to_list()==list(range(16242))
    x=np.zeros((4,len(q),12));seen=np.zeros((4,len(q)),int)
    for (table,),local in q.group_by('table_id'):
        native=int(local['fold'][0]);ix=local['query_id'].to_numpy()
        for f in range(4):
            if native==f:continue
            a,b=sorted([f,native]);path=values.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet'
            assert json.loads(path.with_suffix('.json').read_text())['config']==cfg
            z=local.select('query_id').join(pl.read_parquet(path),on='query_id',validate='1:1',maintain_order='left')
            v=z.select(FIELDS).to_numpy();assert np.isfinite(v).all()
            x[f,ix]=v;seen[f,ix]+=1;x[native,ix]+=v/3;seen[native,ix]+=1
    for f in range(4):
        np.testing.assert_array_equal(seen[f],np.where(q['fold'].to_numpy()==f,3,1))
        np.savez_compressed(ROOT/f'features_fold{f}.npz',x=x[f])
    q.write_parquet(ROOT/'queries.parquet')
    source=[Path(__file__),Path(base.__file__),Path(values.__file__)]
    config=dict(method=__doc__,arms=ARMS,fields=FIELDS,rollout_config=cfg,
                field_note='rollout_config retained for shared verifier compatibility; these are120 conditional-policy values.',
                schedule='Original R33 Cat400D5lr.035L2=8 seeds; donor integration, censoring and three-round isolation MIL unchanged.',
                scope='Two fixed arms; no stopping, family splice, seed or blend sweep. Evaluate half new head recipe plus half grounded62.',
                prior_screen='120 action NLL regressed; that proxy alone did not test evidence retrieval.',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));print('prepared',x.shape,flush=True)
    base.baseline()

def verify():
    import session105_verify as verifier
    verifier.s=base;verifier.main()

if __name__=='__main__':
    setup();{'prepare':prepare,'train':base.train,'verify':verify}[sys.argv[1]]()
