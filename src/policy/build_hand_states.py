\
\
\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','8')
import json,time,numpy as np,polars as pl
from pathlib import Path
C=pl.col; V1=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(exist_ok=True,parents=True); root=Path('artifacts/policy'); S=6
FEAT=['eq0','eq1','eq2','eq3','fold_street','contrib_bb','seat_no','rank_high','rank_low','suited','pocket','net_bb','won_share','showdown','stack_bb']
json.dump(FEAT,open(OUT/'state_columns.json','w'))
hands_all=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','big_blind']); seats_all=pl.read_parquet('data/seats.parquet'); t0=time.time()
for i,p in enumerate(sorted(V1.glob('*.npz'))):
    table=p.stem; out=OUT/f'{table}.npz'
    if out.exists(): continue
    z=np.load(p); a=pl.read_parquet(root/'actions'/f'{table}.parquet').sort('hand_id','action_no')
    players=sorted(set(a['player_id'].to_list())|set(z['player_1'].tolist())|set(z['player_2'].tolist())); pidx={q:k for k,q in enumerate(players)}
    hands=a.select('hand_id','time_index').unique().sort('time_index'); hid2i={h:k for k,h in enumerate(hands['hand_id'])}; H=len(hands)
    hb=hands_all.filter(C('table_id')==table).select('hand_id','big_blind')
    se=seats_all.join(hb,on='hand_id',how='inner').filter(C('hand_id').is_in(list(hid2i)))
    st=pl.read_parquet(root/'states'/f'{table}.parquet').select('hand_id','player_id','street_no','equity','rank_high','rank_low','suited','pocket')
    d=se.select('hand_id','player_id','seat_no',(C('starting_stack')/C('big_blind')).alias('stack_bb'),(C('total_contribution')/C('big_blind')).alias('contrib_bb'),(C('net_chips')/C('big_blind')).alias('net_bb'),'won_share',C('went_to_showdown').cast(pl.Float32).alias('showdown'))
    for s in range(4): d=d.join(st.filter(C('street_no')==s).select('hand_id','player_id',C('equity').alias(f'eq{s}')),on=['hand_id','player_id'],how='left')
    d=d.with_columns(C('eq0').fill_null(0.0)).with_columns(pl.coalesce(['eq1','eq0']).alias('eq1')).with_columns(pl.coalesce(['eq2','eq1']).alias('eq2')).with_columns(pl.coalesce(['eq3','eq2']).alias('eq3'))
    d=d.join(st.filter(C('street_no')==0).select('hand_id','player_id','rank_high','rank_low','suited','pocket'),on=['hand_id','player_id'],how='left')
    fs=a.filter(C('action_class')==0).group_by('hand_id','player_id').agg(C('street_no').min().alias('fold_street'))
    d=d.join(fs,on=['hand_id','player_id'],how='left').with_columns(C('fold_street').fill_null(4)).fill_null(0)
    d=d.with_columns(C('seat_no').rank('ordinal').over('hand_id').alias('slot')-1).filter(C('slot')<S)
    hi=np.array([hid2i[h] for h in d['hand_id']]); si=d['slot'].to_numpy().astype(int)
    ROST=np.full((H,S),-1,np.int16); HS=np.zeros((H,S,len(FEAT)),np.float16)
    ROST[hi,si]=np.array([pidx.get(q,-1) for q in d['player_id']],np.int16); HS[hi,si]=d.select(FEAT).to_numpy().astype(np.float16)
    hidx=np.array([hid2i.get(h,-1) for h in z['hand_id']],np.int32); assert (hidx>=0).all()
    np.savez(out,ROST=ROST,HS=HS,hidx=hidx,players=np.array(players))
    if i%40==0: print(i,table,H,round(time.time()-t0),flush=True)
print('done',round(time.time()-t0),flush=True)
