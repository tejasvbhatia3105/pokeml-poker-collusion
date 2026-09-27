\
\
\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
from session5_multiway_features import equity,CARD
C=pl.col;ROOT=Path('artifacts/evidence_session33_rollout');STREET={'preflop':0,'flop':1,'turn':2,'river':3}
FIELDS=['own_call_edge','partner_fold_gain','own_share','partner_share','pot_odds','extra_fraction']
def action_class(row):
 return 3 if row['amount']>row['to_call'] else {'fold':0,'check':1}.get(row['action'],2)
def table_features(table,query):
 need=query.select('hand_id').unique();raw=pl.read_parquet(f'artifacts/compact/actions/table_id={table}/*.parquet').join(need,on='hand_id',how='semi').sort('hand_id','action_no');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');meta={r['hand_id']:r for r in h.to_dicts()};seats={hid:z.sort('seat_no') for (hid,),z in s.group_by('hand_id')};mapping={}
 for pid,hid,a,b in query.select('pair_id','hand_id','player_1','player_2').iter_rows():
  mapping.setdefault((hid,a),[]).append((pid,b));mapping.setdefault((hid,b),[]).append((pid,a))
 prepared=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet',columns=['hand_id','action_no','action_class']).join(need,on='hand_id',how='semi');classes={(h,int(n)):int(k) for h,n,k in prepared.iter_rows()};states=[];masks=[];lookup={};records=[];checks={'actions':0,'stack_error':0.,'pot_error':0.,'call_error':0.,'refund_layers':0,'sidepot_scenarios':0,'normalized_claim_mass_error':0.,'no_side_claim_difference':0.,'action_class_mismatch':0,'allin_calls':0}
 def register(hid,street,mask):
  assert mask>0;key=(hid,street,mask)
  if key not in lookup:
   ss=seats[hid];cards=sum(([CARD[a],CARD[b]] for a,b in ss.select('hole_card_1','hole_card_2').iter_rows()),[]);nb=0 if street==0 else street+2;cards+=[CARD[c] for c in meta[hid]['board_cards'].split()[:nb]]+[-1]*(5-nb);lookup[key]=len(states);states.append(cards);masks.append(mask)
  return lookup[key]
 def layers(hid,street,con,alive):
  out=[];refund=np.zeros(6);previous=0.
  for level in np.unique(con[con>0]):
   present=con>=level;eligible=present&alive;mass=float((level-previous)*present.sum());previous=level
   if eligible.any():out.append((register(hid,street,sum(1<<j for j in np.flatnonzero(eligible))),mass))
   else:
    assert present.sum()==1;refund[present]+=mass;checks['refund_layers']+=1
  assert abs(sum(w for _,w in out)+refund.sum()-con.sum())<1e-8
  return out,refund
 for (hid,),g in raw.group_by('hand_id',maintain_order=True):
  ss=seats[hid];hh=meta[hid];people=ss['player_id'].to_list();index={p:i for i,p in enumerate(people)};stacks=ss['starting_stack'].to_numpy().astype(float);net=ss['net_chips'].to_numpy();con=np.zeros(6);streetcon=np.zeros(6);alive=np.ones(6,bool);street=0;last_aggressor=None;button=int(hh['button_seat']);bb=hh['big_blind']
  for seat,amount in [((button+1)%6,hh['small_blind']),((button+2)%6,bb)]:
   j=int(np.flatnonzero(ss['seat_no'].to_numpy()==seat)[0]);con[j]=min(stacks[j],amount);streetcon[j]=con[j]
  for row in g.to_dicts():
   st=STREET[row['street']]
   if st!=street:street=st;streetcon[:]=0;last_aggressor=None
   who=row['player_id'];own=index[who];bet=streetcon.max();remaining=stacks-con;call=min(remaining[own],max(0,bet-streetcon[own]));checks['actions']+=1;checks['stack_error']=max(checks['stack_error'],abs(remaining[own]-row['stack_before']));checks['pot_error']=max(checks['pot_error'],abs(con.sum()-row['pot_before']));checks['call_error']=max(checks['call_error'],abs(call-row['to_call']));assert alive[own]
   checks['action_class_mismatch']+=int(action_class(row)!=classes[hid,row['action_no']]);checks['allin_calls']+=int(row['action']=='all_in' and action_class(row)==2);targets=mapping.get((hid,who),[])
   if targets:
    additions=np.minimum(remaining,np.maximum(0,bet-streetcon))*alive;called=con+additions;folded=called.copy();folded[own]=con[own];falive=alive.copy();falive[own]=False;assert falive.any();call_layers,call_refund=layers(hid,street,called,alive);fold_layers,fold_refund=layers(hid,street,folded,falive);fullmask=sum(1<<j for j in np.flatnonzero(alive));foldmask=fullmask&~(1<<own);ci=register(hid,street,fullmask);fi=register(hid,street,foldmask);checks['sidepot_scenarios']+=int(any(masks[i]!=fullmask for i,_ in call_layers));immediate=con.copy();immediate[own]+=call;contestable=float(np.minimum(immediate,immediate[own]).sum());total=float(called.sum());ftotal=float(folded.sum());den=max(total,1.);pot=float(con.sum())
    has_side=any(masks[i]!=fullmask for i,_ in call_layers) or any(masks[i]!=foldmask for i,_ in fold_layers)
    if has_side:assert np.any((stacks-called)[alive]==0)
    for pid,partner in targets:
     other=index[partner];records.append({'pair_id':pid,'hand_id':hid,'action_no':row['action_no'],'own':own,'other':other,'action_class':action_class(row),'lower':bool(net[own]<=net[other]),'higher':bool(net[own]>=net[other]),'partner_alive':bool(alive[other]),'players_active':int(alive.sum()),'call_layers':call_layers,'fold_layers':fold_layers,'call_refund':call_refund,'fold_refund':fold_refund,'ci':ci,'fi':fi,'total':total,'ftotal':ftotal,'call':float(call),'den':den,'pot':pot,'contestable':contestable,'facing_partner':False,'has_side':has_side})
                                                                           
   if targets:
    for rec in records[-len(targets):]:rec['facing_partner']=last_aggressor==people[rec['other']]
   if action_class(row)==3:last_aggressor=who
   con[own]+=row['amount'];streetcon[own]=row['amount_to']
   if row['action']=='fold':alive[own]=False
 assert max(checks['stack_error'],checks['pot_error'],checks['call_error'],checks['action_class_mismatch'])==0
 eq=equity(states,masks);output=[]
 for r in records:
  own=r['own'];other=r['other'];cp=r['call_refund'].copy();fp=r['fold_refund'].copy()
  for i,mass in r['call_layers']:cp+=eq[i]*mass
  for i,mass in r['fold_layers']:fp+=eq[i]*mass
  nc=eq[r['ci']]*r['total'];nf=eq[r['fi']]*r['ftotal'];base=max(r['pot']+r['call'],1.);v={k:r[k] for k in ['pair_id','hand_id','action_no','action_class','lower','higher','partner_alive','players_active','facing_partner']};checks['normalized_claim_mass_error']=max(checks['normalized_claim_mass_error'],abs(cp.sum()-r['total'])/r['den'],abs(fp.sum()-r['ftotal'])/r['den'])
  if not r['has_side']:checks['no_side_claim_difference']=max(checks['no_side_claim_difference'],float(abs(cp-nc).max())/r['den'],float(abs(fp-nf).max())/r['den'])
  for arm,ca,fo in [('naive',nc,nf),('side',cp,fp)]:
   values=[(ca[own]-r['call']-fo[own])/r['den'],(fo[other]-ca[other])/r['den'],ca[own]/r['den'],ca[other]/r['den'],r['call']/max(r['contestable'] if arm=='side' else base,1.),1-r['contestable']/base if arm=='side' else (r['total']-r['pot']-r['call'])/base];v.update({arm+'_'+k:float(x) for k,x in zip(FIELDS,values)})
  output.append(v)
 assert max(checks['normalized_claim_mass_error'],checks['no_side_claim_difference'])<5e-7
 a=pl.DataFrame(output);assert np.isfinite(a.select(*[arm+'_'+k for arm in ['naive','side'] for k in FIELDS]).to_numpy()).all();expr=[]
 for arm in ['naive','side']:
  for role in ['lower','higher']:
   for name,gate in [('partner',C('facing_partner')&C('partner_alive')),('outside',~C('facing_partner')&C('partner_alive')&(C('players_active')>=3)),('hu',C('partner_alive')&(C('players_active')==2))]:
    mask=C(role)&gate
    for field in FIELDS:
     v=C(arm+'_'+field).filter(mask);prefix=f'claims_{arm}_{role}_{name}_{field}';expr.extend([v.mean().fill_null(0).alias(prefix+'_mean'),v.max().fill_null(0).alias(prefix+'_max')])
  for role in ['lower','higher']:
   for name,gate in [('fold_partner',(C('action_class')==0)&C('facing_partner')),('call_partner',(C('action_class')==2)&C('facing_partner')),('raise_outside',(C('action_class')==3)&~C('facing_partner')&(C('players_active')>=3))]:
    mask=C(role)&C('partner_alive')&gate
    for field in FIELDS:
     v=C(arm+'_'+field).filter(mask);prefix=f'claims_{arm}_{role}_{name}_{field}';expr.extend([v.mean().fill_null(0).alias(prefix+'_mean'),v.max().fill_null(0).alias(prefix+'_max')])
 out=a.group_by('pair_id','hand_id').agg(expr).with_columns(pl.selectors.numeric().cast(pl.Float32));checks['pair_member_actions']=len(a);checks['equity_states']=len(states);checks['max_equity_sum_error']=float(abs(eq.sum(1)-1).max());return out,checks
def main():
 (ROOT/'features').mkdir(exist_ok=True);d=hand_data();labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');q=d.select('pair_id','hand_id','table_id').join(labs,on='pair_id',validate='m:1');audit=[];start=time.time();(ROOT/'feature_config.json').write_text(json.dumps({'method':__doc__,'equity':'128 deterministic samples preflop/flop; exhaustive turn; exact river; all dealt cards treated as known dead cards','arms':'naive matched-call scenario versus same scenario plus contribution-aware claims','aggregation':'72 role/context summaries +72 fold/call/raise gated summaries per arm; 144 each; rebuilt before event training'},indent=2))
 for i,((table,),g) in enumerate(q.group_by('table_id')):
  path=ROOT/'features'/f'{table}.parquet'
  if path.exists() and os.environ.get('SIDEPOT_FORCE','0')!='1':info=json.load(open(path.with_suffix('.json')))
  else:
   out,info=table_features(table,g.drop('table_id'));assert out.select('pair_id','hand_id').n_unique()==len(g);out.write_parquet(path);path.with_suffix('.json').write_text(json.dumps(info,indent=2))
  audit.append({'table_id':table,**info})
  if i%30==0:print('sidepot features',i,round(time.time()-start,1),flush=True)
 pl.read_parquet(list((ROOT/'features').glob('T*.parquet'))).write_parquet(ROOT/'hand_features.parquet');(ROOT/'feature_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
