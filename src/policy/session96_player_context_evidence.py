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
from catboost import CatBoostClassifier,CatBoostRegressor
import session95_learned_player_kernel as p
import session83_joint_policy as old
from session37_bet_fold import paired_features
from session84_nested_joint_policy import paths
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session96_player_context');C=pl.col;KINDS=['current','context']
CONFIG={'method':__doc__,'arms':KINDS,'event_schedule':'Cat400 D5 lr.035 L2=8 original head seeds6311+11fold','extra_fields':15,'context_sampling':p.s.CONFIG,'new_context_exclusion':'all shared focal-pair hands, not only query or evidence hands','policy_representation':'frozen outer95 models; evidence-training pools not inner-cross-fitted at meta-policy stage','reference':'strict nested84 action/size policies','fixed_recipe':'unchanged nonprimary heads; equal average with full62 tree; no family/weight selection'}
def models():
 action=p.s.load_models();size={};meta={}
 for key in action:
  m=CatBoostRegressor();m.load_model(str(paths(*key)[1]));size[key]=m
 for f in range(4):
  for k in KINDS:m=CatBoostClassifier();m.load_model(str(p.ROOT/f'{k}_fold{f}.cbm'));meta[f,k]=m
 return action,size,meta
def pair_input(src,keys,shared):
 qq=keys.join(src.with_columns(C('action_no').cast(pl.Int64)),on=['hand_id','action_no'],validate='m:1',maintain_order='left').sort('query_row');assert (qq['player_id']==qq['expected_player']).all();qq=qq.drop('expected_player');people=qq['player_id'].unique();hist=src.filter(~C('hand_id').is_in(pl.Series(list(shared)).implode())&C('player_id').is_in(people.implode()));h=pl.concat([z.sample(n=min(len(z),p.s.CONFIG['context_cap_per_player']),seed=p.s.CONFIG['sample_seed']) for _,z in hist.group_by('player_id',maintain_order=True)]).sort('source_row');assert not len(h.select('hand_id').join(pl.DataFrame({'hand_id':list(shared)}),on='hand_id',how='semi'));return p.s.restyle(hist,qq),p.s.restyle(hist,h)
def policy_features(q,h,native,ms):
 action,size,meta=ms;x=q.select(p.s.PC).to_numpy();hx=h.select(p.s.PC).to_numpy();pr={key:m.predict_proba(x,thread_count=2) for key,m in action.items() if native in key};hr={key:action[key].predict_proba(hx,thread_count=2) for key in pr};sz={key:size[key].predict(x,thread_count=2) for key in pr};out={};role=q['role'].to_numpy();assert np.array_equal(role,np.tile([0,1],len(q)//2))
 for f in range(4):
  key=tuple(sorted([native,f]));pp=p.legal(sum(pr.values())/3,x) if native==f else p.legal(pr[key],x);hp=p.legal(sum(hr.values())/3,hx) if native==f else p.legal(hr[key],hx);sizepred=sum(sz.values())/3 if native==f else sz[key];prior=np.log(pp.clip(1e-7)).astype(np.float32);base=np.column_stack([x,prior]);ex=p.fields(q,h,pp,hp);res=q.filter(C('role')==1)['log_bet_ratio'].to_numpy()-sizepred[role==1]
  for kind in KINDS:
   xx=np.column_stack([base,ex]) if kind=='context' else base;prob=p.predict(meta[f,kind],xx,prior);own,bet=prob[role==0],prob[role==1];u=-np.log(own[:,0].clip(1e-7));v=-np.log(bet[:,3].clip(1e-7));out[f,kind]=np.column_stack([own,bet,u,v,u+v,np.minimum(u,v),u-v,res,abs(res)]).astype(np.float32)
 return out
def prepare():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));ms=models();states=old.state();start=time.time();audit=[]
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];_,align=paired_features(d,a);keys=pl.concat([align.select('action_row','pair_id','hand_id',C('fold_action_no').alias('action_no'),C('actor').alias('expected_player')).with_columns(pl.lit(0).alias('role')),align.select('action_row','pair_id','hand_id',C('bet_action_no').alias('action_no'),C('partner').alias('expected_player')).with_columns(pl.lit(1).alias('role'))]);keys=keys.with_columns((2*C('action_row')+C('role')).alias('query_row'));values={(f,k):np.zeros((len(a),15),np.float32) for f in range(4) for k in KINDS};raw=[];align.write_parquet(ROOT/f'{fam}_alignment.parquet')
  for (table,),group in d.group_by('table_id'):
   src=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');local=keys.join(group.select('pair_id').unique(),on='pair_id',how='semi')
   for (pid,),query in local.group_by('pair_id'):
    pair=group.filter(C('pair_id')==pid);q,h=pair_input(src,query,set(pair['hand_id']));native=int(pair['fold'][0]);pred=policy_features(q,h,native,ms);ix=q.filter(C('role')==0)['action_row'].to_numpy();raw.append(q)
    for fk,z in pred.items():values[fk][ix]=z
    audit.append({'family':fam,'pair_id':pid,'table_id':table,'query_actions':len(q),'context_actions':len(h),'excluded_shared_hands':len(pair),'query_context_overlap':0})
  z=pl.concat(raw).sort('query_row');np.testing.assert_array_equal(z['query_row'].to_numpy(),np.arange(2*len(a)));z.write_parquet(ROOT/f'{fam}_queries.parquet')
  for (f,k),x in values.items():
   dest=ROOT/k;dest.mkdir(exist_ok=True);assert (x[:,:8].sum(1)>1.99).all();np.savez_compressed(dest/f'{fam}_extra_fold{f}.npz',x=x)
  print('player evidence prepared',fam,len(a),round(time.time()-start,1),flush=True)
 (ROOT/'input_audit.json').write_text(json.dumps(audit,indent=2))
def train():
 states=old.state();base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');parts={k:[] for k in KINDS};audit=[];start=time.time()
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2));pred={k:np.zeros((len(d),2)) for k in KINDS}
  for f in range(4):
   if fam=='directed_transfer':y,tr,va=old.labels(d,a,f)
   else:yy,e,_=old.target(d,f);y=yy[g];tr=e[g];va=fv[g]==f
   assert not(tr&va).any()
   for k in KINDS:
    root=ROOT/k;extra=np.load(root/f'{fam}_extra_fold{f}.npz')['x'];x=np.column_stack([v['x'],extra]);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(root/f'{fam}_primary_fold{f}.cbm'));pred[k][g[va],actor[va]]=m.predict_proba(x[va],thread_count=2)[:,1];audit.append({'family':fam,'fold':f,'kind':k,'training_actions':int(tr.sum()),'validation_overlap':0});print('player evidence',fam,f,k,round(time.time()-start,1),flush=True)
  for k,z in pred.items():parts[k].append(d.select('pair_id','hand_id').with_columns(pl.Series('replacement',(z*dw).sum(1))))
 for k in KINDS:
  root=ROOT/k;base.join(pl.concat(parts[k]),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('replacement','bg_primary').alias('bg_primary')).drop('replacement').write_parquet(root/'event_oof.parquet');assemble(root)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 else:train()
