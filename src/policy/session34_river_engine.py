\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
from dataclasses import dataclass
import numpy as np,polars as pl
from session8_data import hand_data
from session33_terminal_replay import payout,ranks
from session33_sidepot_features import action_class
C=pl.col;ROOT=Path('artifacts/evidence_session34_river')
@dataclass
class River:
 starting:np.ndarray
 contribution:np.ndarray
 committed:np.ndarray
 alive:np.ndarray
 pending:np.ndarray
 raise_right:np.ndarray
 minimum_raise:float
 button:int
 cursor:int
 action_no:int
 previous_action:int
 last_aggressor:int=-1
 prior_raises:int=0
 def clone(self):
  return River(self.starting.copy(),self.contribution.copy(),self.committed.copy(),self.alive.copy(),self.pending.copy(),self.raise_right.copy(),self.minimum_raise,self.button,self.cursor,self.action_no,self.previous_action,self.last_aggressor,self.prior_raises)
 @property
 def remaining(self):return self.starting-self.contribution
 @property
 def bet(self):return self.committed.max()
 def clean(self):
  eligible=self.alive&(self.remaining>0);self.pending&=eligible
  if self.alive.sum()<=1:self.pending[:]=False
  if eligible.sum()==1:
   j=np.flatnonzero(eligible)[0]
   if self.committed[j]>=self.bet:self.pending[:]=False
 def next_actor(self):
  self.clean()
  for k in range(1,7):
   j=(self.cursor+k)%6
   if self.pending[j]:return j
  return None
 def to_call(self,j):return min(self.remaining[j],max(0.,self.bet-self.committed[j]))
 def apply(self,j,k,amount):
  call=self.to_call(j);remaining=self.remaining[j];oldbet=self.bet;violations=[]
  if not self.pending[j]:violations.append('actor_not_pending')
  if amount<0 or amount>remaining:violations.append('payment_outside_stack')
  if k==0 and (amount!=0 or call<=0):violations.append('illegal_fold')
  if k==1 and (amount!=0 or call>0):violations.append('illegal_check')
  if k==2 and amount!=call:violations.append('incorrect_call_payment')
  if k==3:
   if amount<=call:violations.append('raise_not_above_call')
   if not self.raise_right[j]:violations.append('raise_not_reopened')
   inc=self.committed[j]+amount-oldbet
   if inc<self.minimum_raise and amount!=remaining:violations.append('non_allin_underraise')
  self.pending[j]=False;self.raise_right[j]=False
  if k==0:self.alive[j]=False
  self.contribution[j]+=amount;self.committed[j]+=amount
  if k==3 and self.committed[j]>oldbet:
   inc=self.committed[j]-oldbet
   if inc>=self.minimum_raise:
    self.minimum_raise=inc;self.pending=self.alive&(self.remaining>0);self.raise_right=self.pending.copy();self.pending[j]=False;self.raise_right[j]=False
   else:self.pending|=self.alive&(self.remaining>0)&(self.committed<self.bet)
   self.last_aggressor=j;self.prior_raises+=1
  self.cursor=j;self.previous_action=k;self.action_no+=1;self.clean();return violations
def initialize(hh,seats,actions):
 seats=seats.sort('seat_no');assert seats['seat_no'].to_list()==list(range(6));people=seats['player_id'].to_list();index={p:i for i,p in enumerate(people)};starting=seats['starting_stack'].to_numpy().astype(float);con=np.zeros(6);alive=np.ones(6,bool);button=int(hh['button_seat'])
 for j,amount in [((button+1)%6,hh['small_blind']),((button+2)%6,hh['big_blind'])]:con[j]=min(starting[j],amount)
 previous=-1
 for row in actions:
  if row['street']=='river':
   pending=alive&(starting>con);return River(starting,con,np.zeros(6),alive,pending,pending.copy(),float(hh['big_blind']),button,button,int(row['action_no']),previous),index
  j=index[row['player_id']];con[j]+=row['amount'];alive[j]&=row['action']!='fold';previous=action_class(row)
 return None,index
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();report={'purpose':__doc__,'river_hands':0,'river_actions':0,'violations':{},'turn_mismatches':0,'unfinished_rounds':0,'stack_error':0.,'pot_error':0.,'call_error':0.,'street_commitment_error':0.,'active_count_error':0,'contribution_error':0.,'payout_error':0.};examples=[];start=time.time()
 for (table,),q in d.group_by('table_id'):
  needed=q.select('hand_id').unique();a=pl.read_parquet(f'artifacts/compact/actions/table_id={table}/*.parquet').join(needed,on='hand_id',how='semi').sort('hand_id','action_no');rivers=a.filter(C('street')=='river').select('hand_id').unique();a=a.join(rivers,on='hand_id',how='semi');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(rivers,on='hand_id',how='semi');s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(rivers,on='hand_id',how='semi');meta={r['hand_id']:r for r in h.to_dicts()};seats={hid:z.sort('seat_no') for (hid,),z in s.group_by('hand_id')}
  for (hid,),g in a.group_by('hand_id'):
   rows=g.to_dicts();ss=seats[hid];st,index=initialize(meta[hid],ss,rows);assert st is not None;report['river_hands']+=1
   for row in rows:
    if row['street']!='river':continue
    j=index[row['player_id']];expected=st.next_actor();report['turn_mismatches']+=int(expected!=j);report['river_actions']+=1
    for name,value in [('stack_error',st.remaining[j]-row['stack_before']),('pot_error',st.contribution.sum()-row['pot_before']),('call_error',st.to_call(j)-row['to_call']),('street_commitment_error',st.committed[j]-(row['amount_to']-row['amount'])),('active_count_error',st.alive.sum()-row['players_active'])]:report[name]=max(report[name],abs(float(value)))
    failures=st.apply(j,action_class(row),row['amount'])
    for name in failures:report['violations'][name]=report['violations'].get(name,0)+1
    if (expected!=j or failures) and len(examples)<30:examples.append({'hand_id':hid,'table':table,'row':row,'expected_actor':expected,'actual_actor':j,'violations':failures})
   unfinished=st.next_actor() is not None;report['unfinished_rounds']+=int(unfinished);report['contribution_error']=max(report['contribution_error'],float(abs(st.contribution-ss['total_contribution'].to_numpy()).max()));gross=payout(st.contribution,st.alive,ranks(ss,meta[hid]['board_cards']));assert gross is not None;report['payout_error']=max(report['payout_error'],float(abs(gross-st.contribution-ss['net_chips'].to_numpy()).max()))
 report['seconds']=time.time()-start;(ROOT/'replay.json').write_text(json.dumps(report,indent=2));(ROOT/'examples.json').write_text(json.dumps(examples,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
