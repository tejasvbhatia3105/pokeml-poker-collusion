import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time
import numpy as np,polars as pl
from cards import CARD,features,preflop_table
ROOT=Path('artifacts/compact');DEST=Path('artifacts/policy/actions');STATES=Path('artifacts/policy/states');DEST.mkdir(exist_ok=True);STATES.mkdir(exist_ok=True)
PF=preflop_table();C=pl.col
STREET={'preflop':0,'flop':1,'turn':2,'river':3}
t=time.time()
for ti,hp in enumerate(sorted((ROOT/'hands').glob('table_id=*'))):
 table=hp.name.split('=')[1];out=DEST/f'{table}.parquet'
 if out.exists() and not os.environ.get('ONLY_STATES'):continue
 h=pl.read_parquet(hp/'*.parquet').sort('started_at').with_row_index('time_index').with_columns((C('time_index')//250).alias('time_bin'))
 s=pl.read_parquet(ROOT/'seats'/hp.name/'*.parquet')
 a=pl.read_parquet(ROOT/'actions'/hp.name/'*.parquet').sort('hand_id','action_no').with_columns(C('street').replace_strict(STREET).alias('street_no'),pl.when(C('amount')>C('to_call')).then(3).when(C('action')=='fold').then(0).when(C('action')=='check').then(1).otherwise(2).alias('action_class'))
 folds=a.group_by('hand_id','player_id').agg(C('action_no').filter(C('action')=='fold').min().fill_null(999).alias('fold_no'))
 s=s.join(folds,on=['hand_id','player_id'],how='left').with_columns(C('fold_no').fill_null(999))
 starts=a.group_by('hand_id','street_no').agg(C('action_no').min().alias('street_start'))
 states=s.join(starts,on='hand_id').join(h.select('hand_id','phase','board_cards','time_index','time_bin','button_seat','big_blind'),on='hand_id')
 states=states.with_columns(C('hole_card_1').replace_strict(CARD).alias('card_1'),C('hole_card_2').replace_strict(CARD).alias('card_2'))
 cards=np.full((len(states),7),-1,np.int8);cards[:,0]=states['card_1'];cards[:,1]=states['card_2'];streets=states['street_no'].to_numpy()
 boards={hid:[CARD[c] for c in board.split()] for hid,board in h.select('hand_id','board_cards').iter_rows()}
 for i,(hid,st) in enumerate(states.select('hand_id','street_no').iter_rows()):
  if st:cards[i,2:st+4]=boards[hid][:st+2]
 rank1=cards[:,0]//4;rank2=cards[:,1]//4;suited=(cards[:,0]%4)==(cards[:,1]%4)
 eq=PF[rank1,rank2,suited.astype(int)];cat=np.zeros(len(states),np.float32);kick=np.zeros(len(states),np.float32)
 post=streets>0
 if post.any():
  z=features(cards[post],96);cat[post]=z[:,0];kick[post]=z[:,1];eq[post]=z[:,2]
                                                 
 br=cards[:,2:]//4+2;bs=cards[:,2:]%4;visible=cards[:,2:]>=0
 board_pairs=np.stack([((br==r)&visible).sum(1) for r in range(2,15)],1).max(1)
 board_suits=np.stack([((bs==r)&visible).sum(1) for r in range(4)],1).max(1)
 board_high=np.where(visible,br,0).max(1)
 same_suit=np.maximum(((bs==cards[:,0,None]%4)&visible).sum(1),((bs==cards[:,1,None]%4)&visible).sum(1))
 hole_matches=(((br==rank1[:,None]+2)|(br==rank2[:,None]+2))&visible).sum(1)
 states=states.with_columns(*[pl.Series(n,v) for n,v in {'equity':eq,'made_category':cat,'made_kicker':kick,'rank_high':np.maximum(rank1,rank2)+2,'rank_low':np.minimum(rank1,rank2)+2,'suited':suited.astype(np.int8),'pocket':(rank1==rank2).astype(np.int8),'board_max_multiplicity':board_pairs,'board_max_suit':board_suits,'board_high':board_high,'hole_suit_matches':same_suit,'hole_board_matches':hole_matches}.items()])
 statecols=['equity','made_category','made_kicker','rank_high','rank_low','suited','pocket','board_max_multiplicity','board_max_suit','board_high','hole_suit_matches','hole_board_matches']
 states.select('hand_id','player_id','street_no','fold_no',*statecols).write_parquet(STATES/f'{table}.parquet',compression='zstd')
 if os.environ.get('ONLY_STATES'):
  if ti%40==0:print('states',ti,round(time.time()-t,1),flush=True)
  continue
 a=a.with_columns(pl.when(C('action_class')==3).then(C('player_id')).otherwise(None).forward_fill().shift(1).over(['hand_id','street']).alias('last_aggressor'),C('action_class').shift(1).over('hand_id').fill_null(-1).alias('previous_action'),(C('action_class')==3).cast(pl.Int32).cum_sum().shift(1).over(['hand_id','street']).fill_null(0).alias('prior_raises'))
 a=a.join(states.select('hand_id','player_id','street_no','phase','time_index','time_bin','button_seat','big_blind','seat_no',*statecols),on=['hand_id','player_id','street_no'])
                                                                           
 keys=['player_id','phase','street_no'];hk=keys+['hand_id','time_bin']
 counts=a.group_by(hk).agg(pl.len().alias('hand_n'),*[(C('action_class')==k).sum().alias(f'hand_{k}') for k in range(4)])
 globalstyle=counts.group_by(keys).agg(C('hand_n').sum().alias('global_n'),*[C(f'hand_{k}').sum().alias(f'global_{k}') for k in range(4)])
 localstyle=counts.group_by(keys+['time_bin']).agg(C('hand_n').sum().alias('local_n'),*[C(f'hand_{k}').sum().alias(f'local_{k}') for k in range(4)])
 counts=counts.join(globalstyle,on=keys).join(localstyle,on=keys+['time_bin'])
 counts=counts.with_columns(*[((C(f'global_{k}')-C(f'hand_{k}')+1)/(C('global_n')-C('hand_n')+4)).alias(f'style_{k}') for k in range(4)])
 counts=counts.with_columns(*[((C(f'local_{k}')-C(f'hand_{k}')+20*C(f'style_{k}'))/(C('local_n')-C('hand_n')+20)).alias(f'local_style_{k}') for k in range(4)])
 styles=[f'{p}_{k}' for p in ['style','local_style'] for k in range(4)]
 a=a.join(counts.select(hk+styles),on=hk)
 a=a.with_columns((C('pot_before')/C('big_blind')).alias('pot_bb'),(C('to_call')/C('big_blind')).alias('call_bb'),(C('stack_before')/C('big_blind')).alias('stack_bb'),(C('to_call')/(C('pot_before')+C('to_call')).clip(1)).alias('pot_odds'),(C('to_call')/C('stack_before').clip(1)).alias('call_stack'),((C('seat_no')-C('button_seat'))%6).alias('position'),(C('last_aggressor')==C('player_id')).fill_null(False).cast(pl.Int8).alias('self_last_aggressor'),((C('amount')/C('big_blind'))+1).log().alias('log_amount_bb'),((C('amount')/C('pot_before').clip(1))+.01).log().alias('log_bet_ratio'))
 cols=statecols+styles+['street_no','action_no','players_active','previous_action','prior_raises','big_blind','pot_bb','call_bb','stack_bb','pot_odds','call_stack','position','self_last_aggressor']
 ids=['hand_id','player_id','phase','time_index','time_bin','last_aggressor','action_class','amount','to_call','log_amount_bb','log_bet_ratio']
 a=a.select(*ids,*[C(c).cast(pl.Float32) for c in cols]).with_columns(pl.lit(table).alias('table_id'))
 a.write_parquet(out,compression='zstd')
 Path('artifacts/policy/feature_columns.json').write_text(json.dumps(cols))
 if ti%20==0:print(ti,len(a),'seconds',round(time.time()-t,1),flush=True)
