\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','8')
import json,time,numpy as np,polars as pl
from pathlib import Path
C=pl.col; ROWS=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(exist_ok=True,parents=True)
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at','big_blind','final_pot','players_at_showdown']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id','net_chips','folded','went_to_showdown','total_contribution','seat_no','starting_stack'])
channels=['alive_agg','alive_fold','partner_call','weak_partner_call','partner_surrender','hu_passivity','hu_check','outsider_agg','weak_outsider_agg','dealt_weak_agg','hidden_agg_alive','hidden_fold_alive','hidden_agg_folded','hidden_call_folded','yield_better','size_partner','size_outsider']
actor1=['surp_sum_1','surp_alive_1','surp_facing_1','n_act_1','n_facing_1','surp_max_1']; actor2=[c.replace('_1','_2') for c in actor1]
t0=time.time()
for i,p in enumerate(sorted(ROWS.glob('*.parquet'))):
    table=p.stem; out=OUT/f'{table}.npz'
    if out.exists(): continue
    h=pl.read_parquet(p)
    st=pl.read_parquet(f'artifacts/policy/states/{table}.parquet').filter(C('street_no')==0).select('hand_id','player_id','equity')
    s1=seats.rename({'player_id':'player_1'}).select('hand_id','player_1',C('net_chips').alias('net1'),C('folded').alias('f1'),C('went_to_showdown').alias('sd1'),C('total_contribution').alias('put1'),C('seat_no').alias('seat1'),C('starting_stack').alias('stk1'))
    s2=seats.rename({'player_id':'player_2'}).select('hand_id','player_2',C('net_chips').alias('net2'),C('folded').alias('f2'),C('went_to_showdown').alias('sd2'),C('total_contribution').alias('put2'),C('seat_no').alias('seat2'),C('starting_stack').alias('stk2'))
    h=h.join(s1,on=['hand_id','player_1']).join(s2,on=['hand_id','player_2']).join(hands.select('hand_id','big_blind','final_pot','players_at_showdown','idx'),on='hand_id')
    h=h.join(st.rename({'player_id':'player_1','equity':'eq1'}),on=['hand_id','player_1'],how='left').join(st.rename({'player_id':'player_2','equity':'eq2'}),on=['hand_id','player_2'],how='left').fill_null(0.5)
    bb=C('big_blind')
    h=h.with_columns(*[(C(n+'_r')/(C(n+'_v')+1).sqrt()).alias(n+'_z') for n in channels],*[(C(n+'_v')+1).log().alias(n+'_lv') for n in channels],
        (C('net1')/bb).clip(-100,100).alias('net1_bb'),(C('net2')/bb).clip(-100,100).alias('net2_bb'),(C('put1')/bb).clip(0,200).log1p().alias('put1_l'),(C('put2')/bb).clip(0,200).log1p().alias('put2_l'),
        C('f1').cast(pl.Float32),C('f2').cast(pl.Float32),C('sd1').cast(pl.Float32),C('sd2').cast(pl.Float32),(C('final_pot')/bb).log1p().alias('pot_l'),((C('seat1')-C('seat2'))%6).cast(pl.Float32).alias('seatd'),
        C('players_at_showdown').cast(pl.Float32),(C('stk1')/bb).log1p().alias('stk1_l'),(C('stk2')/bb).log1p().alias('stk2_l'),
        pl.when(C('phase')=='development').then(C('idx')/3000.0).otherwise((C('idx')-3000)/2000.0).alias('trel'))
    cols1=[n+'_z' for n in channels]+[n+'_lv' for n in channels]+['surprise_max','surp_alive_sum']+actor1+actor2+['net1_bb','net2_bb','put1_l','put2_l','f1','f2','sd1','sd2','pot_l','seatd','players_at_showdown','stk1_l','stk2_l','eq1','eq2','trel']
    h=h.sort('pair_id','phase','time_index')
    X=h.select(cols1).to_numpy().astype(np.float16)
    grp=h.group_by('pair_id','phase',maintain_order=True).agg(pl.len().alias('n'),C('player_1').first(),C('player_2').first())
    np.savez_compressed(out,X=X,hand_id=h['hand_id'].to_numpy().astype('U15'),time_index=h['time_index'].to_numpy().astype(np.int32),pair_id=grp['pair_id'].to_numpy().astype('U27'),phase=grp['phase'].to_numpy().astype('U11'),n=grp['n'].to_numpy().astype(np.int32),player_1=grp['player_1'].to_numpy().astype('U13'),player_2=grp['player_2'].to_numpy().astype('U13'))
    if i==0: json.dump(cols1,open(OUT/'columns.json','w')); print('features',len(cols1),flush=True)
    if i%40==0: print(i,table,X.shape,round(time.time()-t0),flush=True)
print('done',round(time.time()-t0),flush=True)
