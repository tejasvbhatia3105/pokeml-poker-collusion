import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np
import polars as pl
import time
RANK={r:i+2 for i,r in enumerate('23456789TJQKA')}
SUIT={s:i for i,s in enumerate('cdhs')}
STREETS=['preflop','flop','turn','river']

def card_features(s,h):
    s=s.join(h.select('hand_id','board_cards'),on='hand_id')
    cards=s.select('hole_card_1','hole_card_2','board_cards').rows()
    n=len(cards); ranks=np.zeros((n,7),np.int8); suits=np.full((n,7),-1,np.int8)
    for i,(a,b,board) in enumerate(cards):
        for j,c in enumerate([a,b]+board.split()): ranks[i,j]=RANK[c[0]]; suits[i,j]=SUIT[c[1]]
    hi=ranks[:,:2].max(1); lo=ranks[:,:2].min(1); pocket=hi==lo; suited=suits[:,0]==suits[:,1]
    pf=np.clip((hi+lo-4)/24*.65+pocket*.35+suited*.06+(hi-lo<=2)*.04,0,1)
    out={'strength_0':pf,'category_0':pocket.astype(float),'draw_0':suited.astype(float),'high':hi,'low':lo,'pocket':pocket.astype(np.int8),'suited':suited.astype(np.int8)}
    for street,k in enumerate([5,6,7],1):
        rr=ranks[:,:k]; ss=suits[:,:k]; cnt=np.stack([(rr==r).sum(1) for r in range(15)],1); cnt[:,0]=0
        pair=(cnt>=2).sum(1); trip=(cnt>=3).sum(1); quad=(cnt>=4).any(1)
        flush=np.stack([(ss==v).sum(1) for v in range(4)],1).max(1)
        straight=np.zeros(n,bool); draws=np.zeros(n,np.int8)
        present=cnt>0; present[:,1]=present[:,14]
        for v in range(1,11):
            c=present[:,v:v+5].sum(1); straight|=c==5; draws=np.maximum(draws,c)
        cat=np.select([quad,(trip>=1)&(pair>=2),flush>=5,straight,trip>=1,pair>=2,pair>=1],[7,6,5,4,3,2,1],default=0)
                                                                                                    
        made=np.max(np.where(cnt>=2,np.arange(15)[None,:],0),axis=1)
        holematch=((rr[:,2:]==ranks[:,0,None])|(rr[:,2:]==ranks[:,1,None])).sum(1)+pocket
        out[f'strength_{street}']=cat+made/20+hi/400+lo/8000
        out[f'category_{street}']=cat
        out[f'draw_{street}']=(flush>=4).astype(float)+(draws>=4).astype(float)
        out[f'holematch_{street}']=holematch
    return s.drop('board_cards').with_columns([pl.Series(k,v) for k,v in out.items()])

def build(table,labels,ev):
    root=Path(os.environ.get('POKER_PARTITION_ROOT','artifacts/partitioned')); d=f'table_id={table}'
    h=pl.read_parquet(root/'hands'/d/'*.parquet').sort('started_at').with_row_index('time_index')
    s=card_features(pl.read_parquet(root/'seats'/d/'*.parquet'),h)
    a=pl.read_parquet(root/'actions'/d/'*.parquet').sort(['hand_id','action_no'])
    f=a.group_by(['hand_id','player_id']).agg(pl.col('action_no').filter(pl.col('action')=='fold').min().fill_null(999).alias('fold_no'))
    s=s.join(f,on=['hand_id','player_id'],how='left').with_columns(pl.col('fold_no').fill_null(999))
    ids=s['player_id'].unique()
    pairs=pl.concat([labels.filter(pl.col('player_1').is_in(ids.implode())).select('pair_id','player_1','player_2').with_columns(pl.lit('development').alias('phase')),
                     ev.filter(pl.col('player_1').is_in(ids.implode())).select('pair_id','player_1','player_2').with_columns(pl.lit('evaluation').alias('phase'))])
    if not len(pairs):return None
    sh=s.join(h.select('hand_id','phase'),on='hand_id')
    base=sh.join(pairs,left_on=['player_id','phase'],right_on=['player_1','phase']).rename({'player_id':'player_1'})
    base=base.join(s.rename({c:c+'_2' for c in s.columns if c!='hand_id'}),left_on=['hand_id','player_2'],right_on=['hand_id','player_id_2'])
    base=base.join(h.drop('phase'),on='hand_id')
                                                                         
    mapping=pl.concat([base.select('pair_id','hand_id','player_1','player_2').rename({'player_1':'actor','player_2':'partner'}),base.select('pair_id','hand_id','player_2','player_1').rename({'player_2':'actor','player_1':'partner'})])
    a=a.with_columns((pl.col('amount')>pl.col('to_call')).alias('aggressive'))
    a=a.with_columns(pl.when(pl.col('aggressive')).then(pl.col('player_id')).otherwise(None).forward_fill().shift(1).over(['hand_id','street']).alias('last_aggressor'))
    ap=a.join(mapping,left_on=['hand_id','player_id'],right_on=['hand_id','actor'])
    cols=['fold_no']+[f'{x}_{i}' for i in range(4) for x in ['strength','category','draw']]+['high','low','pocket','suited']+[f'holematch_{i}' for i in range(1,4)]
    ap=ap.join(s.select('hand_id','player_id',*cols),on=['hand_id','player_id'])
    ap=ap.join(s.select('hand_id','player_id','fold_no').rename({'player_id':'partner','fold_no':'partner_fold_no'}),on=['hand_id','partner'])
    ap=ap.join(h.select('hand_id','big_blind'),on='hand_id')
    def streetval(prefix):
        x=pl.col(prefix+'_3')
        for i in [2,1,0]:x=pl.when(pl.col('street')==STREETS[i]).then(pl.col(prefix+f'_{i}')).otherwise(x)
        return x
    ap=ap.with_columns(streetval('strength').alias('strength'),streetval('category').alias('category'),streetval('draw').alias('draw'),
        (pl.col('partner_fold_no')>=pl.col('action_no')).alias('alive'),(pl.col('last_aggressor')==pl.col('partner')).fill_null(False).alias('facing'),
        (pl.col('amount')/pl.col('big_blind')).alias('amount_bb'),(pl.col('to_call')/pl.col('big_blind')).alias('call_bb'),
        (pl.col('amount')/pl.col('pot_before').clip(1)).alias('bet_pot'))
    ap=ap.with_columns((pl.col('alive')&(pl.col('players_active')==2)).alias('hu'),
        pl.when(pl.col('street')=='preflop').then(pl.col('strength')<.40).otherwise((pl.col('category')==0)&(pl.col('draw')==0)).alias('weak'),
        pl.when(pl.col('street')=='preflop').then(pl.col('strength')>.72).otherwise(pl.col('category')>=2).alias('strong'))
    C=pl.col; agg=C('aggressive'); alive=C('alive'); facing=C('facing'); hu=C('hu'); weak=C('weak'); strong=C('strong'); call=(C('amount')>0)&~agg; fold=C('action')=='fold'; check=C('action')=='check'
    conditions={
        'act':alive,'hu_act':hu,'face_act':facing,'agg_alive':alive&agg,'agg_out':alive&agg&~facing,'agg_partner':alive&agg&facing,
        'weak_agg_out':alive&agg&~facing&weak,'weak_agg_hu':hu&agg&weak,'weak_call_partner':facing&call&weak,'weak_call_hu':hu&call&weak,
        'strong_fold_partner':facing&fold&strong,'fold_partner':facing&fold,'call_partner':facing&call,'check_hu':hu&check,
        'strong_check_hu':hu&check&strong,'strong_call_partner':facing&call&strong,'strong_passive_hu':hu&strong&~agg,
        'agg_hu':hu&agg,'fold_hu':hu&fold,'weak_agg_alive':alive&weak&agg,'big_call_partner':facing&call&(C('call_bb')>=10),
        'overbet_hu':hu&agg&(C('bet_pot')>.9),'overbet_out':alive&~facing&agg&(C('bet_pot')>.9),
        'fold_free':alive&fold&(C('to_call')==0),'strong_fold_out':alive&~facing&fold&strong,
        'weak_fold_partner':facing&fold&weak,'weak_call_out':alive&~facing&call&weak,
    }
    for st in STREETS:
        on=C('street')==st
        for name,cond in {'agg':agg&alive,'agg_out':agg&alive&~facing,'agg_partner':agg&facing,'check_hu':check&hu,'call_partner':call&facing,'fold_partner':fold&facing,'weak_call_partner':weak&call&facing,'weak_agg_out':weak&agg&alive&~facing}.items():conditions[st+'_'+name]=on&cond
    expr=[cond.cast(pl.Float32).sum().alias(name) for name,cond in conditions.items()]
    for name in ['weak_call_partner','weak_call_hu','agg_partner','agg_out','weak_agg_out','call_partner','fold_partner','strong_fold_partner','agg_hu']:
        cond=conditions[name]
        expr.extend([pl.when(cond).then(C('amount_bb')).otherwise(0).sum().cast(pl.Float32).alias(name+'_money'),pl.when(cond).then(C('call_bb')).otherwise(0).max().cast(pl.Float32).alias(name+'_maxcall')])
    af=ap.group_by('pair_id','hand_id').agg(expr)
    def fl(e,name):return e.cast(pl.Float32).alias(name)
    basic=[fl(C('final_pot')/C('big_blind'),'pot'),fl(C('players_at_showdown'),'showdown_n'),fl(C('board_cards').str.split(' ').list.len(),'board_n'),fl(C('time_index')/5000,'time'),
        fl((C('net_chips')+C('net_chips_2'))/C('big_blind'),'team_net'),fl((C('net_chips')-C('net_chips_2'))/C('big_blind'),'net_direction'),
        fl(pl.min_horizontal(C('total_contribution'),C('total_contribution_2'))/C('big_blind'),'min_contrib'),fl(pl.max_horizontal(C('total_contribution'),C('total_contribution_2'))/C('big_blind'),'max_contrib'),
        fl((C('total_contribution')+C('total_contribution_2'))/C('final_pot'),'pot_fraction'),
        fl(C('went_to_showdown')&C('went_to_showdown_2'),'both_showdown'),fl(C('folded')&C('folded_2'),'both_fold'),fl(~C('folded')&~C('folded_2'),'both_survive'),
        fl(pl.min_horizontal(C('starting_stack'),C('starting_stack_2'))/C('big_blind'),'min_stack'),fl(pl.max_horizontal(C('starting_stack'),C('starting_stack_2'))/C('big_blind'),'max_stack'),
        fl((C('seat_no')-C('seat_no_2')).abs(),'seat_distance')]
    for st in range(4):
        for name in ['strength','category','draw']:
            c=f'{name}_{st}'
            basic.extend([fl(pl.min_horizontal(C(c),C(c+'_2')),c+'_min'),fl(pl.max_horizontal(C(c),C(c+'_2')),c+'_max')])
        basic.append(fl(pl.when(C('net_chips')<C('net_chips_2')).then(C(f'strength_{st}')).otherwise(C(f'strength_{st}_2')),f'loser_strength_{st}'))
    out=base.select('pair_id','hand_id','phase',pl.lit(table).alias('table_id'),*basic).join(af,on=['pair_id','hand_id'],how='left').fill_null(0)
    return out

if __name__=='__main__':
    labels=pl.read_csv('data/development_labels.csv');ev=pl.read_csv('data/evaluation_pairs.csv')
    dest=Path('artifacts/hand_features');dest.mkdir(exist_ok=True)
    tables=sorted(p.name.split('=')[1] for p in Path('artifacts/partitioned/hands').iterdir() if p.is_dir())
    t=time.time()
    for i,table in enumerate(tables):
        path=dest/f'{table}.parquet'
        if path.exists():continue
        out=build(table,labels,ev)
        if out is not None:out.write_parquet(path,compression='zstd')
        if i%10==0:print(i,len(tables),table,len(out) if out is not None else 0,round(time.time()-t,1),flush=True)
