import json,time
import numpy as np
import polars as pl
import session111_native_rollout as n
from session8_data import hand_data
from session33_terminal_replay import ranks,payout

def main():
    d=hand_data();start=time.time();actions=0;hands=0;tables=0
    actor=np.empty(1,np.int32);kind=np.empty(1,np.int32);amount=np.empty(1)
    dynamic=np.zeros((1,13),np.float32);legal=np.zeros((1,4),np.uint8);params=np.zeros((1,4))
    terminal=np.empty((1,6));street_counts=np.zeros(4,int)
                                                                               
    engine=n.f.s.engine
    st=engine.Betting(np.array([1000,125,200,1000,1000,1000.]),np.zeros(6),np.zeros(6),np.ones(6,bool),np.ones(6,bool),np.ones(6,bool),100.,5,5,0,-1,street=3,big_blind=100.)
    ns=n.state(st)[None]
    for j,payment in [(0,100),(1,125),(2,200)]:
        actor[0]=j;kind[0]=3;amount[0]=payment
        assert n.LIB.apply_batch(ns,1,actor,kind,amount)==0 and not st.apply(j,3,payment)
        np.testing.assert_array_equal(ns[0],n.state(st))
    assert ns[0,30]==1 and ns[0,36]==100
    for (table,),q in d.group_by('table_id'):
        raw,meta,seats=engine.table_data(table,q.select('hand_id').unique())
        for (hid,),rows in raw.group_by('hand_id',maintain_order=True):
            ss=seats[hid];st,index=engine.initialize(meta[hid],ss);ns=n.state(st)[None]
            for row in rows.to_dicts():
                expected=st.next_decision();n.LIB.decisions(ns,1,actor);assert actor[0]==expected==index[row['player_id']]
                np.testing.assert_array_equal(ns[0],n.state(st))
                n.LIB.dynamic_features(ns,1,actor,dynamic,legal,params)
                want=n.f.s.inputs.vector(st,int(actor[0]),np.zeros(33,np.float32))[20:]
                np.testing.assert_array_equal(dynamic[0],want)
                j=int(actor[0]);c=st.to_call(j);mask=[c>0,c==0,c>0,st.remaining[j]>c and st.raise_right[j] and np.any(st.alive&(st.remaining>0)&(np.arange(6)!=j))]
                np.testing.assert_array_equal(legal[0],mask)
                np.testing.assert_array_equal(params[0],[c,st.remaining[j],st.contribution.sum(),st.minimum_raise])
                kind[0]=engine.action_class(row);amount[0]=row['amount'];assert n.LIB.apply_batch(ns,1,actor,kind,amount)==0
                assert not st.apply(j,int(kind[0]),float(amount[0]));np.testing.assert_array_equal(ns[0],n.state(st))
                street_counts[st.street]+=1;actions+=1
            assert st.next_decision() is None;n.LIB.decisions(ns,1,actor);assert actor[0]==-1
            np.testing.assert_array_equal(ns[0],n.state(st))
            rr=np.zeros(6,np.uint32) if st.alive.sum()==1 else np.asarray(ranks(ss,meta[hid]['board_cards']),np.uint32)
            assert n.LIB.payouts(ns,1,rr[None],terminal)==0
            want=payout(st.contribution,st.alive,rr)-st.contribution;np.testing.assert_array_equal(terminal[0],want);hands+=1
        tables+=1
        if tables%25==0:print('NATIVE_RAW',tables,hands,actions,time.time()-start,flush=True)
    report=dict(tables=tables,hands=hands,actions=actions,street_actions=street_counts.tolist(),state_error=0,
        dynamic_input_error=0,legal_mask_error=0,terminal_payoff_error=0,cumulative_short_allin_regression=True,
        provenance=n.provenance(),seconds=time.time()-start)
    (n.ROOT/'raw_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
