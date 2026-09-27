\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import json
from pathlib import Path
import numpy as np
import polars as pl
from session197_jev_pilot import canonical, available

ROOT = Path('artifacts/evidence_session201_jev_inference_inputs')
C = pl.col
BASIC = ['relative_time','pot','team_net','net_direction','both_showdown','both_fold','both_survive','seat_distance']


def rebuild(keys):
    h = pl.read_parquet(available('data/hands.parquet'), columns=['hand_id','table_id','started_at','phase','big_blind','final_pot'])
    h = h.sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('time_index'))
    h = h.join(keys.select('hand_id').unique(), on='hand_id', how='semi')
    s = pl.scan_parquet(available('data/seats.parquet')).select('hand_id','player_id','net_chips','went_to_showdown','folded','seat_no')
    s = s.join(h.lazy().select('hand_id'), on='hand_id', how='semi').collect(engine='streaming')
    q = keys.join(h, on='hand_id', validate='m:1')
    for i in [1,2]:
        ss = s.rename({c:(f'player_{i}' if c=='player_id' else c+f'_{i}') for c in s.columns if c!='hand_id'})
        q = q.join(ss, on=['hand_id',f'player_{i}'], validate='m:1')
    time = (C('time_index')/5000).cast(pl.Float32)
    return q.select('pair_id','hand_id',
        pl.when(C('phase')=='development').then(time/.6).otherwise((time-.6)/.4).cast(pl.Float32).alias('relative_time'),
        (C('final_pot')/C('big_blind')).cast(pl.Float32).alias('pot'),
        ((C('net_chips_1')+C('net_chips_2'))/C('big_blind')).cast(pl.Float32).alias('team_net'),
        ((C('net_chips_1')-C('net_chips_2'))/C('big_blind')).cast(pl.Float32).alias('net_direction'),
        (C('went_to_showdown_1')&C('went_to_showdown_2')).cast(pl.Float32).alias('both_showdown'),
        (C('folded_1')&C('folded_2')).cast(pl.Float32).alias('both_fold'),
        (~C('folded_1')&~C('folded_2')).cast(pl.Float32).alias('both_survive'),
        (C('seat_no_1')-C('seat_no_2')).abs().cast(pl.Float32).alias('seat_distance'))


if __name__ == '__main__':
    ROOT.mkdir(exist_ok=True)
    d = canonical()
    dev = d.select('pair_id','hand_id').join(pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'), on='pair_id')
    ev = pl.read_parquet('artifacts/evidence_session199_jev_evaluation/metadata.parquet').select('pair_id','hand_id','player_1','player_2')
    q = rebuild(pl.concat([dev,ev]))
    check = d.select('pair_id','hand_id',*BASIC).join(q, on=['pair_id','hand_id'], suffix='_rebuilt', validate='1:1')
    errors = {c:float(np.abs(check[c].to_numpy().astype(np.float32)-check[c+'_rebuilt'].to_numpy()).max()) for c in BASIC}
    assert all(v==0 for v in errors.values()), errors
    out = ev.select('pair_id','hand_id').join(q, on=['pair_id','hand_id'], validate='1:1')
    out.write_parquet(ROOT/'evaluation_basic.parquet')
    (ROOT/'audit.json').write_text(json.dumps({'public_replay_rows':len(check),'max_abs_error':errors,'evaluation_rows':len(out)},indent=2))
    print((ROOT/'audit.json').read_text())
