\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,time,hashlib
from pathlib import Path
import numpy as np
import polars as pl
import session101_betting_engine as engine
from session102_policy_inputs import PC,Styles,card_fields
from cards import CARD

ROOT=Path('artifacts/evidence_session119_private_partner_data');C=pl.col
PUBLIC=PC+['partner_style_'+c for c in PC[12:20]]+[
    'partner_alive','partner_remaining_bb','partner_contribution_bb','partner_committed_bb','relative_seat',
    'partner_last_aggressor','partner_pending','partner_raise_right','partner_last_action',
    'partner_prior_amount_bb']+[f'{who}_prior_{k}' for who in ['own','partner','others'] for k in range(4)]
PRIVATE=['partner_private_'+PC[i] for i in [0,1,2,3,4,5,6,10,11]]+['partner_equity_minus_own','partner_made_minus_own']

def source():
    b=pl.read_parquet('artifacts/evidence_session115_relationship_data/bags.parquet')
    h=pl.read_parquet('artifacts/evidence_session115_relationship_data/hands.parquet').select('row','pair_id','hand_id')
    return h.join(b.select('pair_id','table_id','player_1','player_2','fold','label','behavior_family'),on='pair_id',validate='m:1')

def build(table,q):
    raw,meta,seats=engine.table_data(table,q.select('hand_id').unique())
    policies=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet');styles=Styles(policies)
    pp=policies.join(q.select('hand_id').unique(),on='hand_id',how='semi')
    values=pp.select(PC).to_numpy();refs={key:v for key,v in zip(pp.select('hand_id','action_no').iter_rows(),values)}
    static={key:v[:12] for key,v in zip(pp.select('hand_id','player_id','street_no').iter_rows(),values)}
    pairs={h:z.to_dicts() for (h,),z in q.group_by('hand_id')};rows=[];pub=[];privkeys=[];needed={};prefix_error=0.;actions=0
    for (hid,),g in raw.group_by('hand_id',maintain_order=True):
        ss=seats[hid];st,index=engine.initialize(meta[hid],ss);people=ss['player_id'].to_list()
        holes=[[CARD[a],CARD[b]] for a,b in ss.select('hole_card_1','hole_card_2').iter_rows()]
        board=np.full(5,-1,np.int8);bc=[CARD[c] for c in meta[hid]['board_cards'].split()];board[:len(bc)]=bc
        hist=np.zeros((6,4),np.float32);last=np.full(6,-1,np.float32);amount=np.zeros(6,np.float32)
        eligible={j:[] for j in range(6)}
        for rel in pairs[hid]:
            a=index[rel['player_1']];b=index[rel['player_2']]
            eligible[a].append((rel,b,0));eligible[b].append((rel,a,1))
        for row in g.to_dicts():
            j=index[row['player_id']];assert st.next_decision()==j
            assert st.street==engine.STREETS[row['street']]
            assert abs(st.contribution.sum()-row['pot_before'])<1e-6 and abs(st.to_call(j)-row['to_call'])<1e-6
            own=refs[hid,row['action_no']];k=engine.action_class(row)
            for rel,partner,role in eligible[j]:
                player=people[partner];key=(hid,player,st.street)
                if key not in static and key not in needed:needed[key]=(holes[partner],board.copy(),st.street)
                context=[st.alive[partner],st.remaining[partner]/st.big_blind,st.contribution[partner]/st.big_blind,
                    st.committed[partner]/st.big_blind,(partner-j)%6,st.last_aggressor==partner,st.pending[partner],
                    st.raise_right[partner],last[partner],amount[partner]/st.big_blind]
                x=np.r_[own,styles.vector(hid,player,st.street),context,hist[j],hist[partner],hist.sum(0)-hist[j]-hist[partner]]
                pub.append(x);privkeys.append(key)
                rows.append(dict(pair_id=rel['pair_id'],hand_id=hid,table_id=table,fold=rel['fold'],label=rel['label'],
                    family=rel['behavior_family'],hand_row=rel['row'],action_no=row['action_no'],actor=role,action_class=k))
            hist[j,k]+=1;last[j]=k;amount[j]+=row['amount'];assert not st.apply(j,k,row['amount']);actions+=1
    if needed:
        keys=list(needed);cf=card_fields([needed[k][0] for k in keys],[needed[k][1] for k in keys],[needed[k][2] for k in keys])
        static.update(zip(keys,cf))
                                                                              
                                                                                
    keys=sorted(set(privkeys));sample=keys[::max(1,len(keys)//32)][:32];hh=[];bb=[];tt=[]
    for hid,player,street in sample:
        ss=seats[hid].filter(C('player_id')==player);hh.append([CARD[ss['hole_card_1'][0]],CARD[ss['hole_card_2'][0]]])
        z=np.full(5,-1,np.int8);bc=[CARD[c] for c in meta[hid]['board_cards'].split()];z[:len(bc)]=bc;bb.append(z);tt.append(street)
    if sample:
        calc=card_fields(hh,bb,tt);expected=np.array([static[k] for k in sample]);prefix_error=float(abs(calc-expected).max());assert prefix_error==0
                                                                      
        for i,street in enumerate(tt):bb[i][0 if street==0 else street+2:]=-1
        assert np.array_equal(calc,card_fields(hh,bb,tt))
    xp=np.array(pub,np.float32);pc=np.array([static[k] for k in privkeys],np.float32)
    private=np.column_stack([pc[:,[0,1,2,3,4,5,6,10,11]],pc[:,0]-xp[:,0],pc[:,1]+pc[:,2]-xp[:,1]-xp[:,2]])
    x=np.column_stack([xp,private]).astype(np.float32);assert x.shape[1]==len(PUBLIC)+len(PRIVATE) and np.isfinite(x).all()
    z=pl.DataFrame(rows);assert z.select('pair_id','hand_id','action_no').n_unique()==len(z)
    return z,x,dict(table=table,action_rows=len(z),raw_actions=actions,hand_rows=len(q),missing_static_states=len(needed),card_prefix_checks=len(sample),card_max_error=prefix_error)

def main():
    ROOT.mkdir(exist_ok=True);d=source();config=dict(method=__doc__,public_columns=PUBLIC,private_columns=PRIVATE,
        features_exclude='pair/player/table/hand IDs, family, relationship label, evidence labels, action target, future board/action/outcome',
        style='Current-hand-excluded phase/street and time-bin gameplay counts; no supervised policy probabilities',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cp=ROOT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config
    else:cp.write_text(json.dumps(config,indent=2))
    start=time.time();audit=[]
    for ti,((table,),q) in enumerate(d.sort('table_id','hand_id','pair_id').group_by('table_id',maintain_order=True)):
        target=ROOT/f'{table}.npz';mp=ROOT/f'{table}.parquet';ap=ROOT/f'{table}.json'
        if target.exists() and mp.exists() and ap.exists():audit.append(json.loads(ap.read_text()));continue
        z,x,info=build(table,q);np.savez_compressed(target,x=x);z.write_parquet(mp);ap.write_text(json.dumps(info,indent=2));audit.append(info)
        if ti%10==0:print('PRIVATE_PARTNER_DATA',ti,table,sum(r['action_rows'] for r in audit),round(time.time()-start,1),flush=True)
    (ROOT/'audit.json').write_text(json.dumps(dict(tables=audit,hand_rows=len(d),action_rows=sum(r['action_rows'] for r in audit),seconds=time.time()-start),indent=2))
    print('complete',time.time()-start,flush=True)

if __name__=='__main__':main()
