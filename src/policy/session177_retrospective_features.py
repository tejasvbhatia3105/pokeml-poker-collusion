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
import session122_family_blind_hands as base
ROOT=Path('artifacts/evidence_session177_retrospective_features');C=pl.col
CARD=['equity','rank_high','rank_low','suited','pocket']
SEAT=['starting_stack','total_contribution','net_chips','folded','went_to_showdown','won_share']
ROLE=['log_start_bb','log_contribution_bb','signed_log_net_bb','log_payout_bb','contribution_stack_fraction','net_stack_fraction','payout_pot_fraction','folded','showdown','won_share']+['preflop_'+c for c in CARD]

def slog(x):return np.sign(x)*np.log1p(np.abs(x))

def build(table,q):
    keys=q.select('pair_id','hand_id','player_1','player_2');hands=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(keys.select('hand_id').unique(),on='hand_id',how='semi')
    seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(keys.select('hand_id').unique(),on='hand_id',how='semi')
    cards=pl.read_parquet(f'artifacts/policy/states/{table}.parquet').filter(C('street_no')==0).select('hand_id','player_id',*CARD)
    seats=seats.join(cards,on=['hand_id','player_id'],validate='1:1');assert seats.select(C(CARD).is_null().any()).to_numpy().sum()==0
    z=keys.join(hands.select('hand_id','big_blind','final_pot','players_at_showdown','board_cards'),on='hand_id',validate='m:1',maintain_order='left')
    for role,player in [('a','player_1'),('b','player_2')]:
        z=z.join(seats.select('hand_id',C('player_id').alias(player),*[C(c).alias(role+'_'+c) for c in SEAT+CARD]),on=['hand_id',player],validate='m:1',maintain_order='left')
    assert len(z)==len(q) and z['a_net_chips'].null_count()==z['b_net_chips'].null_count()==0
    bb=z['big_blind'].to_numpy();pot=z['final_pot'].to_numpy();assert np.all(bb>0) and np.all(pot>0)
    def v(r,c):return z[r+'_'+c].to_numpy().astype(float)
    def actor(r):
        start=v(r,'starting_stack');con=v(r,'total_contribution');net=v(r,'net_chips');payout=con+net
        assert np.all(start>0) and np.all(payout>=0)
        return np.column_stack([np.log1p(start/bb),np.log1p(con/bb),slog(net/bb),np.log1p(payout/bb),con/start,net/start,payout/pot,v(r,'folded'),v(r,'went_to_showdown'),v(r,'won_share'),*[v(r,c) for c in CARD]])
    a,b=actor('a'),actor('b');na,nb=v('a','net_chips'),v('b','net_chips');tie=na==nb
    low=np.where((na<=nb)[:,None],a,b);high=np.where((na>=nb)[:,None],a,b);low[tie]=high[tie]=(a[tie]+b[tie])/2
    con=v('a','total_contribution')+v('b','total_contribution');net=na+nb;payout=con+net
    extra=dict(pair_signed_log_net_bb=slog(net/bb),pair_contribution_pot_fraction=con/pot,pair_payout_pot_fraction=payout/pot,
        pair_log_net_gap_bb=np.log1p(abs(na-nb)/bb),pair_start_stack_ratio=np.minimum(v('a','starting_stack'),v('b','starting_stack'))/np.maximum(v('a','starting_stack'),v('b','starting_stack')),
        pair_both_profit=((na>0)&(nb>0)).astype(float),pair_both_loss=((na<0)&(nb<0)).astype(float),
        pair_fold_count=v('a','folded')+v('b','folded'),pair_showdown_count=v('a','went_to_showdown')+v('b','went_to_showdown'),
        final_log_pot_bb=np.log1p(pot/bb),final_players_at_showdown=z['players_at_showdown'].to_numpy(),final_board_count=np.array([len(x.split()) for x in z['board_cards'].fill_null('')]))
    cols=['final_'+role+'_'+c for role in ['lower_net','higher_net'] for c in ROLE]+list(extra)
    x=np.column_stack([low,high,*extra.values()]).astype(np.float32);assert x.shape==(len(q),len(cols)) and np.isfinite(x).all()
                                                                       
    ledger=seats.group_by('hand_id').agg(C('net_chips').sum().alias('net'),C('total_contribution').sum().alias('contribution'),C('won_share').sum().alias('shares')).join(hands.select('hand_id','final_pot'),on='hand_id',validate='1:1')
    audit=dict(table=table,hands=len(q),net_ties=int(tie.sum()),net_conservation_error=float(ledger['net'].abs().max()),pot_conservation_error=float((ledger['contribution']-ledger['final_pot']).abs().max()),share_total_error=float((ledger['shares']-1).abs().max()))
    return x,cols,audit

def main():
    ROOT.mkdir(exist_ok=True);d=pl.read_parquet(base.ROOT/'hands.parquet');out=None;audit=[];checks=[];start=time.time()
    replay_tables={sorted(d.filter(C('fold')==f)['table_id'].unique())[0] for f in range(4)}
    for ti,table in enumerate(sorted(d['table_id'].unique())):
        q=d.filter(C('table_id')==table);x,cols,a=build(table,q)
        if out is None:out=np.zeros((len(d),len(cols)),np.float32)
        out[q['hand_index'].to_numpy()]=x;audit.append(a)
        if table in replay_tables:
            swap=q.reverse().with_columns(C('player_2').alias('player_1'),C('player_1').alias('player_2'),pl.lit('protected_test').alias('behavior_family'),(1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'))
            xx,cc,_=build(table,swap);assert cols==cc;np.testing.assert_array_equal(xx[::-1],x)
            checks.append(dict(table=table,hands=len(q),row_and_role_reversal_exact=True,label_rank_family_mutations_exact=True))
        if ti%60==0:print('RETROSPECTIVE_FEATURES',ti,round(time.time()-start,1),flush=True)
    assert out.shape==(45129,42);assert all(a['net_conservation_error']==a['pot_conservation_error']==0 for a in audit)
    np.save(ROOT/'extra.npy',out);config=dict(method=__doc__,columns=cols,features=len(cols),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),extra_sha256=hashlib.sha256((ROOT/'extra.npy').read_bytes()).hexdigest(),base_config_sha256=hashlib.sha256((base.ROOT/'config.json').read_bytes()).hexdigest())
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));(ROOT/'verification.json').write_text(json.dumps(dict(tables=audit,checks=checks,rows=len(out),no_fitted_teachers=True),indent=2));print('COMPLETE',out.shape,round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
