\
\
\
\
\
import os,json,time,bisect
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
ROOT=Path('artifacts/evidence_session8');C=pl.col
FIELDS=['action_class','street_no','equity','pot_odds','call_stack','players_active','log_amount_bb','log_bet_ratio','stack_bb','call_bb','pot_bb','prior_raises','self_last_aggressor']
def build(table,query):
    a=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').join(query.select('hand_id').unique(),on='hand_id',how='semi').sort('hand_id','action_no');byhand={h:g.to_dicts() for (h,),g in a.group_by('hand_id')};rows=[]
    for pair in query.to_dicts():
        hand=pair['hand_id'];people=(pair['player_1'],pair['player_2']);acts=byhand[hand];indices={p:[j for j,x in enumerate(acts) if x['player_id']==p] for p in people}
        for i,act in enumerate(acts):
            player=act['player_id']
            if player not in people:continue
            partner=people[1] if player==people[0] else people[0];op=indices[partner];own=indices[player];opos=bisect.bisect_left(op,i);apos=bisect.bisect_left(own,i)
            neighbors={'partner_before':op[opos-1] if opos else -1,'partner_after':op[opos] if opos<len(op) else -1,'actor_before':own[apos-1] if apos else -1,'actor_after':own[apos+1] if apos+1<len(own) else -1,'table_before':i-1 if i else -1,'table_after':i+1 if i+1<len(acts) else -1}
            r={'pair_id':pair['pair_id'],'hand_id':hand,'action_no':act['action_no']}
            for name,j in neighbors.items():
                pre='neighbor_'+name+'_';n=acts[j] if j>=0 else None;r[pre+'exists']=float(n is not None)
                for field in FIELDS:r[pre+field]=float(n[field]) if n else -1.
                r[pre+'gap']=abs(float(n['action_no'])-act['action_no']) if n else -1.;r[pre+'same_street']=float(n['street_no']==act['street_no']) if n else 0.
                r[pre+'is_partner']=float(n['player_id']==partner) if n else 0.
            amount=np.expm1(act['log_amount_bb']);call=act['call_bb'];pot=act['pot_bb'];stack=act['stack_bb'];r['neighbor_amount_call_ratio']=amount/max(call,.1);r['neighbor_raise_pot_ratio']=(amount-call)/max(pot+call,.1);r['neighbor_amount_stack_ratio']=amount/max(stack,.1)
            ratio=amount/max(pot,.1);grid=np.array([.25,1/3,.5,2/3,.75,1,1.25,1.5,2,3]);r['neighbor_standard_size_distance']=float(np.min(np.abs(ratio-grid)));r['neighbor_whole_blind_distance']=float(abs(amount-round(amount)))
            j=neighbors['partner_before']
            if j>=0:
                other=acts[j];oa=np.expm1(other['log_amount_bb']);remaining=other['stack_bb']-oa;r['neighbor_partner_stack_coverage']=remaining/max(stack,.1);r['neighbor_matching_amount_distance']=abs(amount-oa)/max(1,pot);r['neighbor_raise_partner_ratio']=amount/max(oa,.1)
            else:
                r['neighbor_partner_stack_coverage']=-1.;r['neighbor_matching_amount_distance']=-1.;r['neighbor_raise_partner_ratio']=-1.
            following=[v for v in acts[i+1:] if v['street_no']==act['street_no']]
            r['neighbor_outside_folds_after']=sum(v['action_class']==0 and v['player_id'] not in people for v in following);r['neighbor_partner_raises_after']=sum(v['action_class']==3 and v['player_id']==partner for v in following);r['neighbor_partner_actions_before']=opos;r['neighbor_partner_actions_after']=len(op)-opos;rows.append(r)
    z=pl.DataFrame(rows).with_columns(C('action_no').cast(pl.Float32),*[C(c).cast(pl.Float32) for c in rows[0] if c.startswith('neighbor_')]);return z
def main():
    ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet');labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');parts=[];start=time.time()
    for i,((table,),g) in enumerate(ix.group_by('table_id')):
        q=g.select('pair_id','hand_id').join(labs,on='pair_id');r=build(table,q)
        if i==0:
            swapped=build(table,q.rename({'player_1':'player_2','player_2':'player_1'}));assert r.sort('pair_id','hand_id','action_no').equals(swapped.sort('pair_id','hand_id','action_no'))
        parts.append(r)
        if i%60==0:print('neighbor tables',i,'seconds',round(time.time()-start,1),flush=True)
    out=pl.concat(parts);assert out.select('pair_id','hand_id','action_no').n_unique()==len(out);out.write_parquet(ROOT/'neighbor_actions.parquet');(ROOT/'neighbor_columns.json').write_text(json.dumps([c for c in out.columns if c.startswith('neighbor_')],indent=2));print(out.shape,flush=True)
if __name__=='__main__':main()
