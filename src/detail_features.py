import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,time
from features import card_features,STREETS

def build(table,base=None,pairs=None):
    root=Path(os.environ.get('POKER_PARTITION_ROOT','artifacts/partitioned'));d=f'table_id={table}'
    if base is None:base=pl.read_parquet(f'artifacts/hand_features/{table}.parquet')
    h=pl.read_parquet(root/'hands'/d/'*.parquet');s=card_features(pl.read_parquet(root/'seats'/d/'*.parquet'),h)
    a=pl.read_parquet(root/'actions'/d/'*.parquet').sort(['hand_id','action_no'])
    a=a.with_columns((pl.col('amount')>pl.col('to_call')).alias('aggressive'))
    a=a.with_columns(pl.when(pl.col('aggressive')).then(pl.col('player_id')).otherwise(None).forward_fill().shift(1).over(['hand_id','street']).alias('last_aggressor'))
    f=a.group_by('hand_id','player_id').agg(pl.col('action_no').filter(pl.col('action')=='fold').min().fill_null(999).alias('fold_no'))
    s=s.join(f,on=['hand_id','player_id'],how='left').with_columns(pl.col('fold_no').fill_null(999))
    if pairs is None:pairs=pl.concat([pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2')])
    b=base.select('pair_id','hand_id').join(pairs,on='pair_id')
    b=b.join(s.select('hand_id','player_id','strength_0').rename({'player_id':'player_1','strength_0':'s1'}),on=['hand_id','player_1']).join(s.select('hand_id','player_id','strength_0').rename({'player_id':'player_2','strength_0':'s2'}),on=['hand_id','player_2'])
    b=b.with_columns(pl.when(pl.col('s1')<=pl.col('s2')).then(pl.col('player_1')).otherwise(pl.col('player_2')).alias('weak_player'),pl.when(pl.col('s1')<=pl.col('s2')).then(pl.col('player_2')).otherwise(pl.col('player_1')).alias('strong_player'))
    result=base
    for role,other in [('weak','strong'),('strong','weak')]:
        z=b.select('pair_id','hand_id',pl.col(role+'_player').alias('player_id'),pl.col(other+'_player').alias('partner'))
        z=z.join(s,on=['hand_id','player_id']).join(h.select('hand_id','big_blind','button_seat'),on='hand_id')
        attributes=['strength_0','strength_1','strength_2','strength_3','category_1','category_2','category_3','draw_1','draw_2','draw_3','holematch_1','holematch_2','holematch_3','high','low','pocket','suited','fold_no']
        sf=z.select('pair_id','hand_id',*[pl.col(c).cast(pl.Float32).alias(role+'_'+c) for c in attributes],
            (pl.col('net_chips')/pl.col('big_blind')).cast(pl.Float32).alias(role+'_net'),
            ((pl.col('seat_no')-pl.col('button_seat'))%6).cast(pl.Float32).alias(role+'_position'),
            (pl.col('total_contribution')/pl.col('big_blind')).cast(pl.Float32).alias(role+'_contribution'))
        az=a.join(z.select('pair_id','hand_id','player_id','partner','big_blind'),on=['hand_id','player_id'])
        az=az.with_columns((pl.col('action_no').rank('ordinal').over(['pair_id','hand_id','street'])-1).alias('decision'))
        az=az.with_columns(pl.col('action').replace_strict({'fold':0,'check':1,'call':2,'bet':3,'raise':4,'all_in':5}).alias('action_code'),
            (pl.col('last_aggressor')==pl.col('partner')).fill_null(False).cast(pl.Int8).alias('facing_partner'),
            (pl.col('amount')/pl.col('big_blind')).alias('amount_bb'),(pl.col('to_call')/pl.col('big_blind')).alias('call_bb'),
            (pl.col('amount_to')/pl.col('big_blind')).alias('to_bb'),(pl.col('amount')/pl.col('pot_before').clip(1)).alias('pot_ratio'),
            (pl.col('amount_to')/(pl.col('amount_to')-pl.col('amount')+pl.col('to_call')).clip(1)).alias('raise_ratio'))
        expr=[]
        for i,st in enumerate(STREETS):
            for dec in range(2):
                cond=(pl.col('street')==st)&(pl.col('decision')==dec)
                for c in ['action_code','facing_partner','amount_bb','call_bb','to_bb','pot_ratio','raise_ratio','players_active']:
                    expr.append(pl.col(c).filter(cond).first().fill_null(-1).cast(pl.Float32).alias(f'{role}_{st}_{dec}_{c}'))
        af=az.group_by('pair_id','hand_id').agg(expr)
        result=result.join(sf,on=['pair_id','hand_id']).join(af,on=['pair_id','hand_id'],how='left')
    return result.fill_null(-1)

if __name__=='__main__':
    dest=Path('artifacts/detail_features');dest.mkdir(exist_ok=True);t=time.time()
    for i,p in enumerate(sorted(Path('artifacts/hand_features').glob('*.parquet'))):
        if (dest/p.name).exists():continue
        z=build(p.stem);z.write_parquet(dest/p.name,compression='zstd')
        if i%20==0:print(i,round(time.time()-t,1),flush=True)
