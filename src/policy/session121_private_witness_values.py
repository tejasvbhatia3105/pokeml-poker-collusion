import json,hashlib,itertools
from pathlib import Path
import numpy as np
import polars as pl
import session120_private_partner_policy as policy
ROOT=Path('artifacts/evidence_session121_private_witness_values');C=pl.col
FIELDS=['log_normal','log_coordinated','coordination_ratio','normal_entropy','coordinated_entropy','condition_total_variation']

def provenance():
    paths=[Path(__file__),Path(policy.__file__),policy.ROOT/'config.json',policy.data.ROOT/'config.json']
    paths+=sorted(policy.ROOT.glob('*.cbm'))
    assert len(paths)==16
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def fields(p,action):
    p=np.clip(p,1e-9,1);log=np.log(p);a=log[np.arange(len(p)),:,action]
    entropy=-(p*log).sum(2)
    return np.column_stack([a[:,0],a[:,1],a[:,1]-a[:,0],entropy[:,0],entropy[:,1],abs(p[:,1]-p[:,0]).sum(1)/2])

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'tables').mkdir(exist_ok=True)
    q=pl.read_parquet('artifacts/evidence_session104_allstreet_rollout/queries.parquet')
    meta=pl.read_parquet(policy.ROOT/'action_index.parquet')
    keys=['pair_id','hand_id','action_no'];q=q.join(meta.select(*keys,'action_row','action_class'),on=keys,validate='1:1',maintain_order='left')
    assert q['action_row'].null_count()==0 and len(q)==16242
    q.write_parquet(ROOT/'queries.parquet');config=dict(method=__doc__,fields=FIELDS,provenance_sha256=provenance())
    cp=ROOT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:cp.write_text(json.dumps(config,indent=2))
    for f,h in itertools.combinations(range(4),2):
        local=q.filter(C('fold').is_in([f,h]));parts=[]
        for kind in ['public','private']:
            z=pl.read_parquet(policy.ROOT/f'{kind}_exclude{f}{h}.parquet')
            joined=local.select('query_id','action_row','action_class').join(z,on='action_row',validate='1:1',maintain_order='left')
            assert joined['condition0_class0'].null_count()==0
            p=joined.select([f'condition{c}_class{k}' for c in [0,1] for k in range(4)]).to_numpy().reshape(-1,2,4)
            x=fields(p,joined['action_class'].to_numpy())
            parts.extend([pl.Series(kind+'_'+name,x[:,i]) for i,name in enumerate(FIELDS)])
        allvalues=local.select('query_id','table_id').with_columns(parts)
        for (table,),z in allvalues.group_by('table_id'):
            path=ROOT/'tables'/f'{table}_exclude{f}{h}.parquet';z.drop('table_id').write_parquet(path)
            path.with_suffix('.json').write_text(json.dumps({'config':config},indent=2))
    print('complete',len(q),'queries; 735 table/reference files')

if __name__=='__main__':main()
