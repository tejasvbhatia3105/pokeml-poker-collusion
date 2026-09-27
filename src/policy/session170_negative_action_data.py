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
import session153_response_features as response
import session158_generic_multiway_features as multiway
import session163_generic_action_data as positive
ROOT=Path('artifacts/evidence_session170_negative_action_data');C=pl.col

def design(table,local):
    meta=pl.read_parquet(raw.ROOT/f'{table}.parquet').with_row_index('local')
    meta=meta.join(local.select('pair_id','hand_id','hand_index'),on=['pair_id','hand_id'],how='inner',validate='m:1')
    x=np.load(raw.ROOT/f'{table}.npz')['x'][meta['local'].to_numpy()]
    mw=multiway.mw.build(table,local.select('pair_id','hand_id','player_1','player_2'),return_actions=True)
    mw=mw.select('pair_id','hand_id','action_no',*multiway.FIELDS,'amount',C('action_class').alias('actual_class')).with_columns(C('action_no').cast(meta.schema['action_no']))
    z=meta.join(mw,on=['pair_id','hand_id','action_no'],validate='1:1',maintain_order='left')
    assert len(z)==len(meta)==len(mw) and not z['amount'].null_count() and (z['action_class']==z['actual_class']).all()
    z=z.join(local.select('hand_index','phase_position','pair_hand_position','log_pair_hand_count'),on='hand_index',validate='m:1',maintain_order='left')
    xx=np.column_stack([x,response.responses(z['amount'].to_numpy(),x),z.select(multiway.FIELDS).to_numpy(),np.eye(4,dtype=np.float32)[z['action_class'].to_numpy()],z.select('phase_position','pair_hand_position','log_pair_hand_count').to_numpy()]).astype(np.float32)
    assert xx.shape==(len(z),96) and np.isfinite(xx).all()
    return z.select('pair_id','hand_id','table_id','fold','hand_index','action_no','actor','action_class'),xx

def main():
    ROOT.mkdir(exist_ok=True);start=time.time()
    d=raw.source().filter(C('label')==0).sort('pair_id','hand_id').with_row_index('hand_index')
    assert d['pair_id'].n_unique()==1488 and len(d)==178620
    timeline=pl.read_parquet('artifacts/evidence_session115_relationship_data/hands.parquet').select('pair_id','hand_id','time_index')
    d=d.join(timeline,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');chron=np.zeros((len(d),3),np.float32)
    for _,g in d.group_by('pair_id'):
        g=g.sort('time_index','hand_id');ix=g['hand_index'].to_numpy()
        chron[ix]=np.column_stack([g['time_index'].to_numpy()/3000,np.arange(len(g))/max(1,len(g)-1),np.full(len(g),np.log1p(len(g)))])
    d=d.with_columns(*[pl.Series(n,chron[:,i]) for i,n in enumerate(['phase_position','pair_hand_position','log_pair_hand_count'])]);d.write_parquet(ROOT/'hands.parquet')
    tables=sorted(d['table_id'].unique());total=sum(pl.read_parquet(raw.ROOT/f'{t}.parquet',columns=['label']).filter(C('label')==0).height for t in tables)
    assert total==545597
    xout=np.lib.format.open_memmap(ROOT/'x.npy',mode='w+',dtype=np.float32,shape=(total,96));metas=[];audit=[];offset=0
    for ti,table in enumerate(tables):
        q=d.filter(C('table_id')==table);meta,x=design(table,q)
        xout[offset:offset+len(x)]=x;offset+=len(x);metas.append(meta)
        audit.append(dict(table=table,actions=len(x),hands=len(q),class_and_keys_exact=True))
        if ti%50==0:print('NEGATIVE_ACTION_DATA',ti,offset,round(time.time()-start,1),flush=True)
    assert offset==total;xout.flush();del xout
    meta=pl.concat(metas).with_row_index('action_row');assert meta.select('pair_id','hand_id','action_no').n_unique()==total
    assert (np.bincount(meta['hand_index'].to_numpy(),minlength=len(d))>0).all();meta.write_parquet(ROOT/'actions.parquet')
    config=dict(method=__doc__,positive_config=json.load(open(positive.ROOT/'config.json')),actions=total,hands=len(d),pairs=1488,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),x_sha256=hashlib.sha256((ROOT/'x.npy').read_bytes()).hexdigest(),unknown_pairs_used=False)
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
    print('COMPLETE',total,round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
