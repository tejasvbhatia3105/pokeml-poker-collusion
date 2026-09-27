import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
ROOT=Path('artifacts/evidence_session8');C=pl.col
def build(table,hands):
    a=pl.read_parquet(f'artifacts/compact/actions/table_id={table}/*.parquet').join(hands,on='hand_id',how='semi').sort('hand_id','action_no');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet');blind=dict(h.select('hand_id','big_blind').iter_rows());s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet');stacks={(h,p):v for h,p,v in s.select('hand_id','player_id','starting_stack').iter_rows()};out=[];errors=[]
    call_errors=[];capped_calls=0
    for (hid,),g in a.group_by('hand_id'):
        bb=blind[hid];street=None;commit={};bet=bb;minraise=bb;prev=None
        for r in g.to_dicts():
            if street!=r['street']:street=r['street'];commit={};bet=bb if street=='preflop' else 0;minraise=bb
            who=r['player_id'];amt=r['amount'];prior=r['amount_to']-amt;call=r['to_call'];pot=r['pot_before'];stack=r['stack_before'];extra=amt-call;agg=extra>0
            if who in commit:errors.append(abs(prior-commit[who]))
            inferred_bet=prior+call;allin=amt==stack;oldmin=minraise
            expected_call=min(stack,max(0,bet-prior))
            call_errors.append(abs(call-expected_call));capped_calls+=int(bet-prior>stack)
            z={'hand_id':hid,'action_no':float(r['action_no']),'ledger_street_investment_bb':prior/bb,'ledger_total_investment_bb':(stacks[hid,who]-stack)/bb,'ledger_street_investment_pot':prior/max(pot,1),'ledger_total_investment_pot':(stacks[hid,who]-stack)/max(pot,1),'ledger_amount_to_bb':r['amount_to']/bb,'ledger_current_bet_bb':inferred_bet/bb,'ledger_min_raise_bb':oldmin/bb,'ledger_raise_increment_bb':max(0,extra)/bb,'ledger_raise_min_ratio':max(0,extra)/max(oldmin,1),'ledger_minimum_raise':float(agg and extra==oldmin),'ledger_under_raise':float(agg and extra<oldmin),'ledger_under_raise_not_allin':float(agg and extra<oldmin and not allin),'ledger_allin':float(allin),'ledger_bet_discrepancy_bb':(inferred_bet-bet)/bb,'ledger_pot_discrepancy_bb':(pot-prev['pot_before']-prev['amount'])/bb if prev else 0.,'ledger_stack_after_pot':(stack-amt)/max(pot+amt,1),'ledger_call_fraction_invested':call/max(prior+call,1),'ledger_extra_fraction_pot':max(0,extra)/max(pot+call,1),'ledger_round_blind_distance':abs(amt/bb-round(amt/bb))}
            for denom,dv in [('pot',pot),('pot_call',pot+call),('stack',stack)]:
                grid=np.array([.25,1/3,.5,2/3,.75,1,1.25,1.5,2,3])*dv;distance=np.min(np.abs(amt-np.floor(grid)));z['ledger_'+denom+'_formula_exact']=float(distance==0);z['ledger_'+denom+'_formula_distance_bb']=distance/bb
            out.append(z);commit[who]=r['amount_to'];bet=max(bet,r['amount_to'])
            if agg and extra>=minraise:minraise=extra
            prev=r
    z=pl.DataFrame(out).with_columns(pl.selectors.numeric().cast(pl.Float32));return z,{'table_id':table,'repeated_actor_decisions':len(errors),'commitment_mismatch_count':int(np.count_nonzero(errors)),'commitment_max_error':max(errors,default=0),'call_mismatch_count':int(np.count_nonzero(call_errors)),'call_max_error':max(call_errors,default=0),'stack_capped_calls':capped_calls}
def aggregate(actions,ledger):
    d=actions.select('pair_id','hand_id','action_no','action_class','facing_partner','mw_alive','mw_lower','mw_higher').join(ledger,on=['hand_id','action_no'],validate='m:1',maintain_order='left');cols=[c for c in ledger.columns if c.startswith('ledger_')];expr=[]
    for k in range(4):
        for facing in (True,False):
            use=(C('action_class')==k)&(C('facing_partner')==facing)&C('mw_alive');prefix=f'ledger_k{k}_face{int(facing)}_'
            for c in cols:expr.append(C(c).filter(use).mean().fill_null(-2).alias(prefix+c))
    return d.group_by('pair_id','hand_id').agg(expr).with_columns(pl.selectors.numeric().cast(pl.Float32))
def main():
    ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet');parts=[];audit=[];start=time.time()
    for i,((table,),g) in enumerate(ix.group_by('table_id')):
        z,a=build(table,g.select('hand_id').unique());parts.append(z);audit.append(a)
        if i%80==0:print('ledger table',i,'seconds',round(time.time()-start,1),flush=True)
    raw=pl.concat(parts);raw.write_parquet(ROOT/'ledger_actions.parquet');aggregate(pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet'),raw).write_parquet(ROOT/'ledger_hand_features.parquet');(ROOT/'ledger_audit.json').write_text(json.dumps(audit,indent=2));print({'actions':len(raw),'commitment_mismatches':sum(x['commitment_mismatch_count'] for x in audit)},flush=True)
if __name__=='__main__':main()
