\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,time
from pathlib import Path
import numpy as np
import polars as pl
from cards import CARD,features,preflop_table
import session101_betting_engine as engine
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session102_allstreet_inputs')
PC=json.load(open('artifacts/policy/feature_columns.json'));INDEX={c:i for i,c in enumerate(PC)}
PF=preflop_table();C=pl.col

def card_fields(holes,boards,streets):
    holes=np.asarray(holes,np.int8);boards=np.asarray(boards,np.int8);streets=np.asarray(streets,int)
    cards=np.full((len(holes),7),-1,np.int8);cards[:,:2]=holes
    for st in range(1,4):cards[streets==st,2:st+4]=boards[streets==st,:st+2]
    r1=cards[:,0]//4;r2=cards[:,1]//4;suited=cards[:,0]%4==cards[:,1]%4
    eq=PF[r1,r2,suited.astype(int)].copy();cat=np.zeros(len(cards),np.float32);kick=cat.copy();post=streets>0
    if post.any():
        z=features(cards[post],96);cat[post]=z[:,0];kick[post]=z[:,1];eq[post]=z[:,2]
    br=cards[:,2:]//4+2;bs=cards[:,2:]%4;visible=cards[:,2:]>=0
    bp=np.stack([((br==r)&visible).sum(1) for r in range(2,15)],1).max(1)
    suits=np.stack([((bs==s)&visible).sum(1) for s in range(4)],1).max(1)
    high=np.where(visible,br,0).max(1)
    same=np.maximum(((bs==cards[:,0,None]%4)&visible).sum(1),((bs==cards[:,1,None]%4)&visible).sum(1))
    matches=(((br==r1[:,None]+2)|(br==r2[:,None]+2))&visible).sum(1)
    return np.column_stack([eq,cat,kick,np.maximum(r1,r2)+2,np.minimum(r1,r2)+2,
                            suited,r1==r2,bp,suits,high,same,matches]).astype(np.float32)

class Styles:
    def __init__(self,p):
        self.where={h:(phase,int(t)) for h,phase,t in p.select('hand_id','phase','time_bin').unique().iter_rows()}
        self.tables=[]
        for keys in [['player_id','phase','street_no'],['player_id','phase','street_no','time_bin'],['hand_id','player_id','street_no']]:
            q=p.group_by(keys).agg(*[(C('action_class')==k).sum().alias(f'n{k}') for k in range(4)])
            self.tables.append({tuple(row[:len(keys)]):np.array(row[len(keys):],np.float64) for row in q.iter_rows()})
        self.cache={}

    def vector(self,hid,player,street):
        key=hid,player,street
        if key not in self.cache:
            phase,bin=self.where[hid];zero=np.zeros(4)
            hand=self.tables[2].get(key,zero)
            global_counts=self.tables[0].get((player,phase,street),zero)-hand
            local=self.tables[1].get((player,phase,street,bin),zero)-hand
            assert global_counts.min()>=0 and local.min()>=0
            g=(global_counts+1)/(global_counts.sum()+4)
            l=(local+20*g)/(local.sum()+20)
            self.cache[key]=np.r_[g,l].astype(np.float32)
        return self.cache[key]

def vector(st,j,template):
    x=template.copy();call=st.to_call(j);stack=st.remaining[j];pot=st.contribution.sum();bb=st.big_blind
    values={'street_no':st.street,'action_no':st.action_no,'players_active':st.alive.sum(),
            'previous_action':st.previous_action,'prior_raises':st.prior_raises,'big_blind':bb,
            'pot_bb':pot/bb,'call_bb':call/bb,'stack_bb':stack/bb,'pot_odds':call/max(pot+call,1),
            'call_stack':call/max(stack,1),'position':(j-st.button)%6,
            'self_last_aggressor':int(st.last_aggressor==j)}
    for key,value in values.items():x[INDEX[key]]=value
    return x

def future_templates(hid,people,holes,visible,street,styles,reps=16):
    \
    holes=np.asarray(holes,np.int8);visible=np.asarray(visible,np.int8)
    assert len(visible)==(0 if street==0 else street+2)
    known=np.r_[holes.ravel(),visible];assert len(np.unique(known))==len(known)
    remaining=np.setdiff1d(np.arange(52),known)
    rng=np.random.default_rng(10201)
    boards=np.empty((reps,5),np.int8);boards[:,:len(visible)]=visible
    for i in range(reps):boards[i,len(visible):]=rng.choice(remaining,5-len(visible),replace=False)
    out=np.zeros((reps,4,6,len(PC)),np.float32)
    for st in range(street,4):
        h=np.tile(holes,(reps,1));b=np.repeat(boards,6,axis=0)
        out[:,st,:,:12]=card_fields(h,b,np.full(len(h),st)).reshape(reps,6,12)
        out[:,st,:,12:20]=np.stack([styles.vector(hid,p,st) for p in people])[None]
    return boards,out

def audit():
    ROOT.mkdir(exist_ok=True);d=hand_data();report={'raw_actions':0,'static_card_samples':0,'max_card_error':0.,
       'max_style_error':0.,'max_dynamic_error':0.,'future_template_cases':0,'future_board_violations':0,
       'current_hand_style_mutation_error':0.,'replicate_prefix_error':0.,'future_observed_fields_used':False};start=time.time()
    for ti,((table,),q) in enumerate(d.group_by('table_id')):
        need=q.select('hand_id').unique();raw,meta,seats=engine.table_data(table,need)
        p=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet');styles=Styles(p)
        pp=p.join(need,on='hand_id',how='semi').sort('hand_id','action_no')
        refs={(h,int(n)):v for (h,n),v in zip(pp.select('hand_id','action_no').iter_rows(),pp.select(PC).to_numpy())}
        sample=pp.unique(['hand_id','player_id','street_no']).sort('hand_id','player_id','street_no').sample(n=min(64,pp.unique(['hand_id','player_id','street_no']).height),seed=102)
        holes={(h,p):[CARD[a],CARD[b]] for h,ss in seats.items() for p,a,b in ss.select('player_id','hole_card_1','hole_card_2').iter_rows()}
        b=np.full((len(sample),5),-1,np.int8);hs=[];sts=[]
        for i,(hid,player,st) in enumerate(sample.select('hand_id','player_id','street_no').iter_rows()):
            board=[CARD[c] for c in meta[hid]['board_cards'].split()];b[i,:len(board)]=board;hs.append(holes[hid,player]);sts.append(int(st))
        cf=card_fields(hs,b,sts);expected=sample.select(PC[:12]).to_numpy()
        report['max_card_error']=max(report['max_card_error'],float(abs(cf-expected).max()));report['static_card_samples']+=len(sample)
        for (hid,),g in raw.group_by('hand_id',maintain_order=True):
            st,index=engine.initialize(meta[hid],seats[hid])
            for row in g.to_dicts():
                j=index[row['player_id']];assert st.next_decision()==j
                expected=refs[hid,row['action_no']];template=np.zeros(len(PC),np.float32);template[:12]=expected[:12]
                template[12:20]=styles.vector(hid,row['player_id'],st.street)
                report['max_style_error']=max(report['max_style_error'],float(abs(template[12:20]-expected[12:20]).max()))
                actual=vector(st,j,template);report['max_dynamic_error']=max(report['max_dynamic_error'],float(abs(actual-expected).max()))
                assert not st.apply(j,engine.action_class(row),row['amount']);report['raw_actions']+=1
        if ti%20==0:
                                                                              
            hid=sorted(seats)[0];ss=seats[hid];people=ss['player_id'].to_list();h=np.array([holes[hid,p] for p in people])
            boards,templates=future_templates(hid,people,h,[],0,styles,4)
            bb,tt=future_templates(hid,people,h,[],0,styles,8)
            np.testing.assert_array_equal(boards,bb[:4]);np.testing.assert_array_equal(templates,tt[:4])
            assert np.isfinite(templates).all()
            for board in boards:report['future_board_violations']+=int(len(np.unique(np.r_[h.ravel(),board]))!=17)
            mut=p.with_columns(pl.when(C('hand_id')==hid).then((C('action_class')+1)%4).otherwise(C('action_class')).alias('action_class'))
            alt=Styles(mut)
            for player in people:
                for street in range(4):report['current_hand_style_mutation_error']=max(report['current_hand_style_mutation_error'],float(abs(styles.vector(hid,player,street)-alt.vector(hid,player,street)).max()))
            report['future_template_cases']+=1
        if ti%50==0:print('INPUT_REPLAY',ti,report['raw_actions'],'seconds',time.time()-start,flush=True)
    report['seconds']=time.time()-start;(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    for k in ['max_card_error','max_style_error','max_dynamic_error','future_board_violations','current_hand_style_mutation_error','replicate_prefix_error']:assert report[k]==0,(k,report[k])

if __name__=='__main__':audit()
