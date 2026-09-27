\
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
import session5_multiway_features as mw
import session122_family_blind_hands as base

ROOT=Path('artifacts/evidence_session158_generic_multiway_features');C=pl.col
FIELDS=['mw_own','mw_partner','mw_team','mw_call_edge','mw_information_gap','mw_partner_fold_gain','mw_fold_value']

def aggregate(q,raw):
    x=raw.select(FIELDS).to_numpy();bag=raw['hand_index'].to_numpy();cls=raw['action_class'].to_numpy();post=raw['street_no'].to_numpy()>0
    masks=[np.ones(len(x),bool),cls==0,(cls==0)&post,cls==2,(cls==2)&post,cls==3,(cls==3)&post]
    parts=[];cols=[]
    for name,mask in zip(base.GROUPS,masks):
        block,_=base.aggregate(x,bag,mask,len(q));parts.append(block[:,1:]);cols += [name+'_'+stat+'_generic_'+c for stat in ['mean','min','max'] for c in FIELDS]
    return np.column_stack(parts),cols

def main():
    ROOT.mkdir(exist_ok=True);d=pl.read_parquet(base.ROOT/'hands.parquet');players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
    out=np.zeros((len(d),147),np.float32);audit=[];start=time.time()
    for ti,table in enumerate(sorted(d['table_id'].unique())):
        local=d.filter(C('table_id')==table).sort('hand_index').rename({'hand_index':'global_hand_index'}).with_row_index('hand_index')
        query=local.select('pair_id','hand_id').join(players,on='pair_id',validate='m:1')
        raw=mw.build(table,query,return_actions=True)
        raw=raw.join(local.select('pair_id','hand_id','hand_index'),on=['pair_id','hand_id'],validate='m:1')
        assert raw.select('pair_id','hand_id','action_no').n_unique()==len(raw)
        x,cols=aggregate(local,raw);assert x.shape==(len(local),147) and np.isfinite(x).all()
                                                                                  
        checks=0
        for i in np.random.default_rng(15801).choice(len(local),min(4,len(local)),replace=False):
            g=raw.filter(C('hand_index')==i)
            masks=[pl.lit(True),C('action_class')==0,(C('action_class')==0)&(C('street_no')>0),C('action_class')==2,(C('action_class')==2)&(C('street_no')>0),C('action_class')==3,(C('action_class')==3)&(C('street_no')>0)]
            expected=[]
            for mask in masks:
                a=g.filter(mask).select(FIELDS).to_numpy()
                expected.extend(np.r_[a.mean(0,dtype=np.float64),a.min(0),a.max(0)] if len(a) else np.full(21,-2))
            np.testing.assert_allclose(x[i],expected,rtol=1e-6,atol=1e-5);checks+=7
        out[local['global_hand_index'].to_numpy()]=x
        audit.append(dict(table=table,hands=len(local),actions=len(raw),scalar_group_checks=checks))
        if ti%40==0:print('GENERIC_MULTIWAY_FEATURES',ti,table,round(time.time()-start,1),flush=True)
    assert out.shape==(45129,147)
    np.save(ROOT/'extra.npy',out)
    config=dict(method=__doc__,columns=cols,rows=len(d),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        calculator_source_sha256=hashlib.sha256(Path(mw.__file__).read_bytes()).hexdigest(),
        base_config_sha256=hashlib.sha256((base.ROOT/'config.json').read_bytes()).hexdigest(),extra_sha256=hashlib.sha256((ROOT/'extra.npy').read_bytes()).hexdigest(),
        excluded_fields=['mw_lower','mw_higher','net_chips','behavior_family','evidence_rank','relationship_label'])
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));(ROOT/'verification.json').write_text(json.dumps(dict(tables=audit,features=147,actions=sum(a['actions'] for a in audit)),indent=2));print('COMPLETE',len(audit),round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
