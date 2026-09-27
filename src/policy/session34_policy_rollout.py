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
from catboost import CatBoostClassifier
from session8_data import hand_data
from session34_river_engine import initialize,action_class
from session34_policy_state import table_data,vector,COLS
from session33_terminal_replay import payout,ranks
C=pl.col;ROOT=Path('artifacts/evidence_session34_river');FIELDS=['own_loss','partner_gain','team_loss','transfer','own_stderr','partner_stderr']
def roots_for_table(table,query):
 a,meta,seats,templates,_=table_data(table,query.select('hand_id').unique());mapping={}
 for pid,hid,p1,p2 in query.select('pair_id','hand_id','player_1','player_2').iter_rows():
  mapping.setdefault((hid,p1),[]).append((pid,p2));mapping.setdefault((hid,p2),[]).append((pid,p1))
 roots=[]
 for (hid,),g in a.group_by('hand_id'):
  rows=g.to_dicts();ss=seats[hid];st,index=initialize(meta[hid],ss,rows);people=ss['player_id'].to_list();rank=ranks(ss,meta[hid]['board_cards']);tm=np.stack([templates.get((hid,p),np.zeros(len(COLS),np.float32)) for p in people]);net=ss['net_chips'].to_numpy()
  for row in rows:
   if row['street']!='river':continue
   j=index[row['player_id']];assert st.next_actor()==j;partners=mapping.get((hid,row['player_id']),[])
   if partners:roots.append({'hand_id':hid,'action_no':row['action_no'],'state':st.clone(),'own':j,'forced_class':action_class(row),'forced_amount':row['amount'],'templates':tm,'ranks':rank,'bb':meta[hid]['big_blind'],'pot':float(st.contribution.sum()),'net':net,'partners':[(pid,index[p]) for pid,p in partners]})
   assert not st.apply(j,action_class(row),row['amount'])
 return roots
def simulate(roots,policy,sizer,meta,reps=32):
 if not roots:return np.zeros((0,4,reps,6)),{'trajectories':0,'simulated_actions':0,'max_steps':0,'max_net_sum_error':0.}
 steps=128;u=np.random.default_rng(3434).random((reps,steps,2));states=[];ri=[];sc=[];rep=[]
 for i,r in enumerate(roots):
  for s in range(4):
   for k in range(reps):states.append(r['state'].clone());ri.append(i);sc.append(s);rep.append(k)
 ri=np.array(ri);sc=np.array(sc);rep=np.array(rep);centers=np.array(meta['centers']);total_actions=0;maxstep=0
 for step in range(steps):
  ids=[];actors=[]
  for i,st in enumerate(states):
   j=st.next_actor()
   if j is not None:ids.append(i);actors.append(j)
  if not ids:break
  ids=np.array(ids);actors=np.array(actors);maxstep=step+1;xx=np.stack([vector(states[i],int(j),roots[ri[i]]['templates'][j],roots[ri[i]]['bb']) for i,j in zip(ids,actors)]);kinds=np.where(xx[:,COLS.index('call_bb')]>0,2,1);amounts=np.array([states[i].to_call(int(j)) for i,j in zip(ids,actors)]);forced=(step==0)&np.isin(sc[ids],[0,2]);normal=(sc[ids]<2)|((step==0)&(sc[ids]==3));normal&=~forced;nn=np.flatnonzero(normal)
  if len(nn):
   pp=policy.predict_proba(xx[nn],thread_count=2);legal=np.ones_like(pp,bool)
   for ii,n in enumerate(nn):
    i=ids[n];j=int(actors[n]);st=states[i];call=st.to_call(j);legal[ii,0]=call>0;legal[ii,1]=call==0;legal[ii,2]=call>0;legal[ii,3]=st.remaining[j]>call and st.raise_right[j] and np.any(st.alive&(st.remaining>0)&(np.arange(6)!=j))
   pp=np.maximum(pp,1e-12)*legal;pp/=pp.sum(1)[:,None];kk=(u[rep[ids[nn]],step,0,None]>np.cumsum(pp,axis=1)).sum(1).clip(0,3);kinds[nn]=kk;amounts[nn]=np.where(kk==2,amounts[nn],0);raises=nn[kk==3]
   if len(raises):
    sp=sizer.predict_proba(xx[raises],thread_count=2);chosen=(u[rep[ids[raises]],step,1,None]>np.cumsum(sp,axis=1)).sum(1).clip(0,sp.shape[1]-1)
    for n,k in zip(raises,chosen):
     i=ids[n];j=int(actors[n]);st=states[i];remaining=st.remaining[j];want=remaining if k==len(centers)-1 else np.floor(max(0.,np.exp(centers[k])-.01)*max(st.contribution.sum(),1.));amounts[n]=min(remaining,max(want,st.to_call(j)+st.minimum_raise))
  for n in np.flatnonzero(forced):kinds[n]=roots[ri[ids[n]]]['forced_class'];amounts[n]=roots[ri[ids[n]]]['forced_amount']
  for i,j,k,amount in zip(ids,actors,kinds,amounts):
   errors=states[i].apply(int(j),int(k),float(amount));assert not errors,(errors,int(k),float(amount))
  total_actions+=len(ids)
 else:raise RuntimeError('River rollout reached 128 actions; no truncated values exported')
 result=np.zeros((len(roots),4,reps,6));error=0.
 for i,st in enumerate(states):
  assert st.next_actor() is None;gross=payout(st.contribution,st.alive,roots[ri[i]]['ranks']);assert gross is not None;net=gross-st.contribution;error=max(error,float(abs(net.sum())));result[ri[i],sc[i],rep[i]]=net
 assert error<1e-8;return result,{'trajectories':len(states),'simulated_actions':total_actions,'max_steps':maxstep,'max_net_sum_error':error}
def aggregate(roots,values,query):
 rows=[]
 for i,r in enumerate(roots):
  own=r['own'];st=r['state'];den=max(1.,r['pot'])
  for pid,partner in r['partners']:
   row={'pair_id':pid,'hand_id':r['hand_id'],'action_no':r['action_no'],'lower':bool(r['net'][own]<=r['net'][partner]),'higher':bool(r['net'][own]>=r['net'][partner]),'partner_alive':bool(st.alive[partner]),'facing_partner':st.last_aggressor==partner,'players_active':int(st.alive.sum()),'action_class':r['forced_class']}
   for arm,a,b in [('learned',0,1),('checkcall',2,3)]:
    difference=values[i,a]-values[i,b];mean=difference.mean(0)/den;stderr=difference.std(0,ddof=1)/np.sqrt(values.shape[2])/den;ownloss=-mean[own];gain=mean[partner];v=[ownloss,gain,-mean[own]-mean[partner],max(0.,ownloss)*max(0.,gain),stderr[own],stderr[partner]];row.update({arm+'_'+k:float(vv) for k,vv in zip(FIELDS,v)})
   rows.append(row)
 if not rows:return None
 a=pl.DataFrame(rows);expr=[]
 for arm in ['checkcall','learned']:
  for role in ['lower','higher']:
   contexts={'partner':C('facing_partner'),'outside':~C('facing_partner')&(C('players_active')>=3),'hu':C('players_active')==2,'fold_partner':(C('action_class')==0)&C('facing_partner'),'call_partner':(C('action_class')==2)&C('facing_partner'),'raise_outside':(C('action_class')==3)&~C('facing_partner')&(C('players_active')>=3)}
   for name,gate in contexts.items():
    mask=C(role)&C('partner_alive')&gate
    for field in FIELDS:
     z=C(arm+'_'+field).filter(mask);prefix=f'rollout_{arm}_{role}_{name}_{field}';expr.extend([z.mean().fill_null(0).alias(prefix+'_mean'),z.max().fill_null(0).alias(prefix+'_max')])
 return a.group_by('pair_id','hand_id').agg(expr).with_columns(pl.selectors.numeric().cast(pl.Float32))
def main():
 (ROOT/'features').mkdir(exist_ok=True);d=hand_data();labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');q=d.select('pair_id','hand_id','table_id').join(labs,on='pair_id',validate='m:1');folds=json.load(open('artifacts/policy/table_folds.json'));policies=[];sizes=[];metadata=[]
 for f in range(4):
  m=CatBoostClassifier();m.load_model(f'artifacts/policy/action_fold{f}.cbm');policies.append(m);s=CatBoostClassifier();s.load_model(f'artifacts/evidence_session22_size_density/density_fold{f}.cbm');sizes.append(s);metadata.append(json.load(open(f'artifacts/evidence_session22_size_density/metadata_fold{f}.json')))
 config={'method':__doc__,'replicates':32,'random_numbers':'same fixed seed3434 replicate/step grid for every action and both response scenarios; no ID-dependent or order-dependent seeds','policy':'original ordinary action model excluding input pool; no competition labels','size':'session22 categorical size model excluding pool; sample bin then use training-bin log-size center, clip to legal min-raise and stack; separate all-in atom','physics':'river engine and policy-input replay exact on 22779 actions; standard sidepots','comparison':'forced observed focal action vs sampled normal focal action; after that learned responses versus check/call control','max_actions':128,'caveat':'learned-model counterfactual, not causal identification; known actual river cards used retrospectively'};(ROOT/'rollout_config.json').write_text(json.dumps(config,indent=2));audit=[];start=time.time();pilot=os.environ.get('RIVER_PILOT','0')=='1'
 for i,((table,),g) in enumerate(q.group_by('table_id')):
  path=ROOT/'features'/f'{table}.parquet'
  if path.exists():audit.append(json.load(open(path.with_suffix('.json'))));continue
  roots=roots_for_table(table,g.drop('table_id'));f=folds[table];value,info=simulate(roots,policies[f],sizes[f],metadata[f]);out=aggregate(roots,value,g);info.update({'table_id':table,'root_actions':len(roots),'pool_fold':f})
  if pilot:
   subset=roots[:3];x,_=simulate(subset,policies[f],sizes[f],metadata[f],reps=8);rev,_=simulate(subset[::-1],policies[f],sizes[f],metadata[f],reps=8);error=float(abs(x-rev[::-1]).max()) if len(subset) else 0.;assert error==0;info['root_permutation_error']=error;info['pilot_seconds']=time.time()-start;(ROOT/'pilot.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2));return
  if out is not None:out.write_parquet(path);path.with_suffix('.json').write_text(json.dumps(info,indent=2))
  audit.append(info)
  if i%15==0:print('river rollout',i,round(time.time()-start,1),'actions',sum(x['root_actions'] for x in audit),flush=True)
 features=pl.read_parquet(list((ROOT/'features').glob('T*.parquet')));out=d.select('pair_id','hand_id').join(features,on=['pair_id','hand_id'],how='left',validate='1:1').fill_null(0);assert len(out.columns)==290 and np.isfinite(out.select(pl.selectors.numeric()).to_numpy()).all();out.write_parquet(ROOT/'hand_features.parquet');(ROOT/'rollout_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
