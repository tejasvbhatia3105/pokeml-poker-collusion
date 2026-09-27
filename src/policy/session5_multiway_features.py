import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import ctypes,json,time
import numpy as np,polars as pl
ROOT=Path('artifacts/evidence_session5');ROOT.mkdir(exist_ok=True)
CARD={r+s:4*i+j for i,r in enumerate('23456789TJQKA') for j,s in enumerate('cdhs')}
LIB=ctypes.CDLL(str((ROOT/'multiway.dylib').resolve()))
LIB.poker_multiway.argtypes=[ctypes.POINTER(ctypes.c_int8),ctypes.POINTER(ctypes.c_uint8),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
def equity(states,masks):
    states=np.ascontiguousarray(states,dtype=np.int8);masks=np.ascontiguousarray(masks,dtype=np.uint8)
    assert states.shape==(len(masks),17) and np.all(masks>0)
    out=np.empty((len(states),6),np.float32)
    LIB.poker_multiway(states.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),len(states),128,out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    assert np.isfinite(out).all() and np.allclose(out.sum(1),1,atol=1e-6)
    return out
def build(table,query,return_actions=False,only_partner_folds=False):
    C=pl.col;root=Path('artifacts/policy');needed=query.select('hand_id').unique()
    a=pl.read_parquet(root/'actions'/f'{table}.parquet').join(needed,on='hand_id',how='semi')
    s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(needed,on='hand_id',how='semi').sort('hand_id','seat_no')
    h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(needed,on='hand_id',how='semi')
    folds={(hid,p):int(no) for hid,p,no in a.filter(C('action_class')==0).select('hand_id','player_id','action_no').iter_rows()}
    boards={hid:[CARD[c] for c in board.split()] for hid,board in h.select('hand_id','board_cards').iter_rows()}
    rosters={};player={};net={}
    for (hid,),g in s.group_by('hand_id'):
        cs=[];people=g['player_id'].to_list()
        for p,c1,c2,v in g.select('player_id','hole_card_1','hole_card_2','net_chips').iter_rows():
            player[hid,p]=len(cs)//2;net[hid,p]=v;cs.extend([CARD[c1],CARD[c2]])
        rosters[hid]=(cs,people)
    mapping=pl.concat([query.select('pair_id','hand_id',C('player_1').alias('player_id'),C('player_2').alias('partner')),
                       query.select('pair_id','hand_id',C('player_2').alias('player_id'),C('player_1').alias('partner'))])
    z=a.join(mapping,on=['hand_id','player_id']).sort('pair_id','hand_id','action_no');states=[];masks=[];indices={};meta=[]
    if only_partner_folds:
        assert return_actions
        z=z.filter((C('action_class')==0)&(C('last_aggressor')==C('partner')).fill_null(False))
        if not len(z):
            return z.with_columns(*[pl.lit(0.,dtype=pl.Float32).alias(c) for c in ['mw_own','mw_partner','mw_partner_fold_gain','mw_team','mw_call_edge','mw_information_gap','mw_fold_value']],*[pl.lit(False).alias(c) for c in ['mw_alive','mw_lower','mw_higher','facing_partner']])
    def register(hid,street,mask):
        key=(hid,street,mask)
        if key not in indices:
            c=rosters[hid][0]+[-1]*5;nb=0 if street==0 else street+2;c[12:12+nb]=boards[hid][:nb]
            assert len(c)==17 and len(set(x for x in c if x>=0))==12+nb
            indices[key]=len(states);states.append(c);masks.append(mask)
        return indices[key]
    for r in z.select('hand_id','player_id','partner','street_no','action_no','action_class','last_aggressor').iter_rows(named=True):
        hid=r['hand_id'];st=int(r['street_no']);p=r['player_id'];q=r['partner'];pa=player[hid,p];pq=player[hid,q]
        mask=sum(1<<j for j,who in enumerate(rosters[hid][1]) if folds.get((hid,who),999)>=r['action_no'])
        assert mask&(1<<pa)
        before=register(hid,st,mask);after=register(hid,st,mask&~(1<<pa)) if mask&~(1<<pa) else before
        meta.append((before,after,pa,pq,bool(mask&(1<<pq)),net[hid,p]<=net[hid,q],net[hid,p]>=net[hid,q]))
    eq=equity(states,masks);m=np.asarray(meta);b=m[:,0].astype(int);aft=m[:,1].astype(int);pa=m[:,2].astype(int);pq=m[:,3].astype(int)
    own=eq[b,pa];partner=eq[b,pq];gain=eq[aft,pq]-partner
    z=z.with_columns(pl.Series('mw_own',own),pl.Series('mw_partner',partner),pl.Series('mw_partner_fold_gain',gain),
        pl.Series('mw_team',own+partner),pl.Series('mw_alive',m[:,4].astype(bool)),pl.Series('mw_lower',m[:,5].astype(bool)),pl.Series('mw_higher',m[:,6].astype(bool)))
    z=z.with_columns((C('mw_own')-C('pot_odds')).alias('mw_call_edge'),(C('mw_own')-C('equity')).alias('mw_information_gap'),
        (C('mw_partner_fold_gain')*C('pot_bb').log1p()).alias('mw_fold_value'))
    if return_actions:
        return z.with_columns((C('last_aggressor')==C('partner')).fill_null(False).alias('facing_partner'))
    fields=['mw_own','mw_partner','mw_team','mw_call_edge','mw_information_gap','mw_partner_fold_gain','mw_fold_value']
    expr=[]
    for role in ['lower','higher']:
        for context,gate in [('partner',(C('last_aggressor')==C('partner')).fill_null(False)&C('mw_alive')),('outside',~(C('last_aggressor')==C('partner')).fill_null(False)&C('mw_alive'))]:
            for act in range(4):
                use=C('mw_'+role)&gate&(C('action_class')==act)
                for field in fields:
                    expr.append(C(field).filter(use).mean().fill_null(-2).alias(f'multi_{role}_{context}_{act}_{field}'))
    return z.group_by('pair_id','hand_id').agg(expr).with_columns(pl.selectors.numeric().cast(pl.Float32))
if __name__=='__main__':
    d=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet');labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');parts=[];t=time.time()
    for i,((table,),q) in enumerate(d.group_by('table_id')):
        parts.append(build(table,q.select('pair_id','hand_id').join(labs,on='pair_id')))
        if i%40==0:print('multiway table',i,'seconds',round(time.time()-t,1),flush=True)
    out=pl.concat(parts);out.write_parquet(ROOT/'multiway_features.parquet');print(out.shape,round(time.time()-t,1),flush=True)
