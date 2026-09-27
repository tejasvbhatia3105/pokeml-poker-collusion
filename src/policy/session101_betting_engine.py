\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
from dataclasses import dataclass,replace
from pathlib import Path
import json,time
import numpy as np
import polars as pl
from session34_river_engine import River,action_class
from session33_terminal_replay import payout,ranks
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session101_allstreet_engine')
C=pl.col
STREETS={'preflop':0,'flop':1,'turn':2,'river':3}

@dataclass
class Betting(River):
    street:int=0
    big_blind:float=2.

    def clone(self):
        return replace(self,**{k:getattr(self,k).copy() for k in
            ['starting','contribution','committed','alive','pending','raise_right']})

    def apply(self,j,k,amount):
        errors=super().apply(j,k,amount)
        if k==3:
                                                                          
                                                                                
            self.raise_right |= self.alive&(self.remaining>0)&((self.bet-self.committed)>=self.minimum_raise)
        return errors

    def advance(self):
        assert self.next_actor() is None
        if self.street==3 or self.alive.sum()<=1 or np.sum(self.alive&(self.remaining>0))<=1:
            return False
        self.street+=1
        self.committed[:]=0
        self.minimum_raise=self.big_blind
        self.pending=self.alive&(self.remaining>0)
        self.raise_right=self.pending.copy()
        self.cursor=self.button
        self.last_aggressor=-1
        self.prior_raises=0
        return True

    def next_decision(self):
        j=self.next_actor()
        while j is None and self.advance():j=self.next_actor()
        return j

def initialize(meta,seats):
    seats=seats.sort('seat_no')
    assert seats['seat_no'].to_list()==list(range(6))
    starting=seats['starting_stack'].to_numpy().astype(np.float64)
    contribution=np.zeros(6)
    button=int(meta['button_seat']);bb=float(meta['big_blind'])
    for j,amount in [((button+1)%6,meta['small_blind']),((button+2)%6,bb)]:
        contribution[j]=min(starting[j],amount)
    alive=np.ones(6,bool);pending=starting>contribution
    state=Betting(starting,contribution,contribution.copy(),alive,pending,pending.copy(),
        bb,button,(button+2)%6,0,-1,big_blind=bb)
    return state,{p:j for j,p in enumerate(seats['player_id'])}

def table_data(table,needed):
    def read(kind):
        return pl.read_parquet(f'artifacts/compact/{kind}/table_id={table}/*.parquet').join(needed,on='hand_id',how='semi')
    raw=read('actions').sort('hand_id','action_no')
    meta={r['hand_id']:r for r in read('hands').to_dicts()}
    seats={hid:z.sort('seat_no') for (hid,),z in read('seats').group_by('hand_id')}
    return raw,meta,seats

def audit():
    ROOT.mkdir(exist_ok=True);d=hand_data();records=[];examples=[];start=time.time()
    test=Betting(np.array([1000,125,200,1000,1000,1000.]),np.zeros(6),np.zeros(6),
        np.ones(6,bool),np.ones(6,bool),np.ones(6,bool),100.,5,5,0,-1,street=3,big_blind=100.)
    assert not test.apply(0,3,100)
    assert not test.apply(1,3,125) and not test.raise_right[0]
    assert not test.apply(2,3,200) and test.raise_right[0] and test.minimum_raise==100
    for (table,),q in d.group_by('table_id'):
        raw,meta,seats=table_data(table,q.select('hand_id').unique())
        policies=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').join(raw.select('hand_id').unique(),on='hand_id',how='semi')
        ref={(hid,int(n)):(int(k),int(prev),int(prior)) for hid,n,k,prev,prior in
             policies.select('hand_id','action_no','action_class','previous_action','prior_raises').iter_rows()}
        r={'table_id':table,'hands':0,'actions':0,'street_actions':[0]*4,
           'violations':{},'turn_errors':0,'street_errors':0,'unfinished_hands':0,
           'state_max_error':0.,'contribution_max_error':0.,'payout_max_error':0.,
           'payout_undefined':0,'policy_history_errors':0,'clone_errors':0}
        for (hid,),rows in raw.group_by('hand_id',maintain_order=True):
            ss=seats[hid];st,index=initialize(meta[hid],ss);r['hands']+=1
            for row in rows.to_dicts():
                expected=st.next_decision();j=index[row['player_id']];street=STREETS[row['street']]
                r['turn_errors']+=int(expected!=j);r['street_errors']+=int(st.street!=street)
                errors=[st.remaining[j]-row['stack_before'],st.contribution.sum()-row['pot_before'],
                        st.to_call(j)-row['to_call'],st.committed[j]-(row['amount_to']-row['amount']),
                        st.alive.sum()-row['players_active'],st.action_no-row['action_no']]
                r['state_max_error']=max(r['state_max_error'],float(np.abs(errors).max()))
                k,prev,prior=ref[hid,row['action_no']]
                r['policy_history_errors']+=int((k,prev,prior)!=(action_class(row),st.previous_action,st.prior_raises))
                if r['actions']%97==0:
                    cloned=st.clone();before=st.contribution.copy();cloned.apply(j,k,row['amount'])
                    r['clone_errors']+=int(not np.array_equal(st.contribution,before))
                violations=st.apply(j,k,row['amount'])
                for error in violations:r['violations'][error]=r['violations'].get(error,0)+1
                if (violations or expected!=j or st.street!=street or max(abs(v) for v in errors)>0) and len(examples)<20:
                    examples.append({'table_id':table,'hand_id':hid,'row':row,'errors':errors,'violations':violations,'expected_actor':expected,'street':st.street})
                r['actions']+=1;r['street_actions'][street]+=1
            r['unfinished_hands']+=int(st.next_decision() is not None)
            r['contribution_max_error']=max(r['contribution_max_error'],float(abs(st.contribution-ss['total_contribution'].to_numpy()).max()))
            rank=np.zeros(6) if st.alive.sum()==1 else ranks(ss,meta[hid]['board_cards'])
            gross=payout(st.contribution,st.alive,rank)
            if gross is None:r['payout_undefined']+=1
            else:r['payout_max_error']=max(r['payout_max_error'],float(abs(gross-st.contribution-ss['net_chips'].to_numpy()).max()))
        records.append(r)
        if len(records)%50==0:print('REPLAY',len(records),'actions',sum(z['actions'] for z in records),'seconds',time.time()-start,flush=True)
    (ROOT/'records.json').write_text(json.dumps(records,indent=2));(ROOT/'examples.json').write_text(json.dumps(examples,indent=2))
    totals={k:sum(r[k] for r in records) for k in ['hands','actions','turn_errors','street_errors','unfinished_hands','payout_undefined','policy_history_errors','clone_errors']}
    totals.update({k:max(r[k] for r in records) for k in ['state_max_error','contribution_max_error','payout_max_error']})
    totals.update(street_actions=np.sum([r['street_actions'] for r in records],axis=0).tolist(),violations={k:sum(r['violations'].get(k,0) for r in records) for k in {k for r in records for k in r['violations']}},seconds=time.time()-start,
        cumulative_short_allin_regression_passed=True,raise_reopening_reference='https://www.pokertda.com/view-poker-tda-rules/ rule47')
    (ROOT/'replay.json').write_text(json.dumps(totals,indent=2));print(json.dumps(totals,indent=2))
    assert not totals['violations']
    for k in ['turn_errors','street_errors','unfinished_hands','payout_undefined','policy_history_errors','clone_errors','state_max_error','contribution_max_error']:assert totals[k]==0,(k,totals[k])
    assert totals['payout_max_error']<=2                                                                                           

if __name__=='__main__':audit()
