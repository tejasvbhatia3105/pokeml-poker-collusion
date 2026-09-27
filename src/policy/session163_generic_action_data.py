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
import session122_family_blind_hands as base
import session153_response_features as response
import session158_generic_multiway_features as multiway

ROOT=Path('artifacts/evidence_session163_generic_action_data');C=pl.col

def main():
    ROOT.mkdir(exist_ok=True);d=pl.read_parquet(base.ROOT/'hands.parquet');bx=np.load(base.ROOT/'x.npy',mmap_mode='r')
    players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');arrays=[];metas=[];audit=[];start=time.time()
    for ti,table in enumerate(sorted(d['table_id'].unique())):
        local=d.filter(C('table_id')==table)
        meta=pl.read_parquet(raw.ROOT/f'{table}.parquet').with_row_index('local')
        meta=meta.join(local.select('pair_id','hand_id','hand_index'),on=['pair_id','hand_id'],validate='m:1',how='inner')
        x=np.load(raw.ROOT/f'{table}.npz')['x'][meta['local'].to_numpy()]
        query=local.select('pair_id','hand_id').join(players,on='pair_id',validate='m:1')
        mw=multiway.mw.build(table,query,return_actions=True)
        keys=['pair_id','hand_id','action_no']
        mw=mw.select(*keys,*multiway.FIELDS,'amount',C('action_class').alias('actual_class')).with_columns(C('action_no').cast(meta.schema['action_no']))
        z=meta.join(mw,on=keys,validate='1:1',maintain_order='left')
        assert len(z)==len(meta)==len(mw) and not z['amount'].null_count() and (z['action_class']==z['actual_class']).all()
        extra=response.responses(z['amount'].to_numpy(),x)
        xx=np.column_stack([x,extra,z.select(multiway.FIELDS).to_numpy(),np.eye(4,dtype=np.float32)[z['action_class'].to_numpy()],bx[z['hand_index'].to_numpy(),-3:]]).astype(np.float32)
        assert xx.shape[1]==96 and np.isfinite(xx).all()
        arrays.append(xx);metas.append(meta.select('pair_id','hand_id','table_id','fold','hand_index','action_no','actor','action_class'))
        audit.append(dict(table=table,hands=len(local),actions=len(meta),observed_class_alignment=True))
        if ti%50==0:print('GENERIC_ACTION_DATA',ti,round(time.time()-start,1),flush=True)
    m=pl.concat(metas);x=np.concatenate(arrays);order=np.lexsort((m['action_no'].to_numpy(),m['hand_index'].to_numpy()))
    m=m[order].with_row_index('action_row');x=x[order]
    assert len(x)==159748 and m.select('pair_id','hand_id','action_no').n_unique()==len(m)
    assert np.all(np.bincount(m['hand_index'].to_numpy(),minlength=len(d))>0)
    cols=raw.PUBLIC+raw.PRIVATE+['response_'+c for c in response.FIELDS]+multiway.FIELDS+[f'observed_class_{i}' for i in range(4)]+['phase_position','pair_hand_position','log_pair_hand_count']
    assert len(cols)==96 and len(set(cols))==96
    np.save(ROOT/'x.npy',x);m.write_parquet(ROOT/'actions.parquet')
    config=dict(method=__doc__,columns=cols,features=96,actions=len(m),hands=len(d),teacher_predictions=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),x_sha256=hashlib.sha256((ROOT/'x.npy').read_bytes()).hexdigest(),
        component_source_sha256={str(Path(module.__file__)):hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in [raw,response,multiway,multiway.mw]},
        limitations='Chronology and hand-excluded phase style summaries are retained from122; not a real-time-only policy predictor. No final chip outcomes or future board ranks enter the96 features.')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));(ROOT/'verification.json').write_text(json.dumps(dict(tables=audit,all_hands_have_actions=True,unique_action_keys=True),indent=2));print('COMPLETE',x.shape,round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
