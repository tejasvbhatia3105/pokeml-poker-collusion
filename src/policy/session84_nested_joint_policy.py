\
\
\
\
\
\
\
import os,json,time,itertools
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
import session83_joint_policy as old
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session84_nested_joint_policy');C=pl.col
def paths(f,h):
 a,b=sorted([f,h]);return ROOT/f'action_exclude{a}{b}.cbm',ROOT/f'size_exclude{a}{b}.cbm'
def reference_data():
 pc=json.load(open('artifacts/policy/feature_columns.json'));tf=json.load(open('artifacts/policy/table_folds.json'));parts=[]
 for path in sorted(Path('artifacts/policy/actions').glob('*.parquet')):
  d=pl.read_parquet(path).filter(C('phase')=='development');parts.append(pl.concat([q.sample(n=min(n,len(q)),seed=414) for street,n in enumerate([600,300,150,150]) if len(q:=d.filter(C('street_no')==street))]).select('table_id','hand_id',*pc,'action_class','log_bet_ratio'))
 d=pl.concat(parts);return d,pc,np.array([tf[t] for t in d['table_id']])
def extra(own,bet,size,f,h,pc):
 ap,sp=paths(f,h);m=CatBoostClassifier();m.load_model(str(ap));p=m.predict_proba(own,thread_count=3);q=m.predict_proba(bet,thread_count=3)
 for z,x in [(p,own),(q,bet)]:
  call=x[:,pc.index('call_bb')]>0;z[call,1]=0;z[~call,0]=0;z[~call,2]=0;z/=z.sum(1)[:,None]
 m=CatBoostRegressor();m.load_model(str(sp));res=size-m.predict(bet,thread_count=3);u=-np.log(p[:,0].clip(1e-7));v=-np.log(q[:,3].clip(1e-7));return np.column_stack([p,q,u,v,u+v,np.minimum(u,v),u-v,res,abs(res)]).astype(np.float32)
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'evidence_head_schedule':'original Cat400 D5 lr.035 L2=8 seed6311+11fold','ordinary_action_schedule':'Cat450 D6 lr.065 L2=10 seed310','ordinary_size_schedule':'Cat350 D6 lr.07 L2=10 Huber.5 seed510','training_reference_fold_exclusion':'outer fold plus each row native fold','validation_reference':'mean of three outer-excluded models, all fitted on two folds','limitations':['repeated evidence validation','ordinary gameplay can include anomalies','existing style/context features retained']},indent=2));start=time.time();d,pc,g=reference_data();x=d.select(pc).to_numpy();y=d['action_class'].to_numpy();size=d['log_bet_ratio'].to_numpy();reports=[]
 for f,h in itertools.combinations(range(4),2):
  tr=(g!=f)&(g!=h);ap,sp=paths(f,h);m=CatBoostClassifier(iterations=450,depth=6,learning_rate=.065,l2_leaf_reg=10,loss_function='MultiClass',random_seed=310,thread_count=3,allow_writing_files=False,verbose=False);m.fit(x[tr],y[tr]);m.save_model(str(ap));m=CatBoostRegressor(iterations=350,depth=6,learning_rate=.07,l2_leaf_reg=10,loss_function='Huber:delta=0.5',random_seed=510,thread_count=3,allow_writing_files=False,verbose=False);rt=tr&(y==3);m.fit(x[rt],size[rt]);m.save_model(str(sp));reports.append({'excluded_folds':[f,h],'training_actions':int(tr.sum()),'training_raises':int(rt.sum()),'training_tables':sorted(d.filter(pl.Series(tr))['table_id'].unique().to_list()),'excluded_action_overlap':0});(ROOT/'reference_audit.json').write_text(json.dumps(reports,indent=2));print('nested reference',f,h,round(time.time()-start,1),flush=True)
 del d,x,y,size,g
 states=old.state();base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');parts=[];audit=[];tf=json.load(open('artifacts/policy/table_folds.json'))
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];raw=np.load(old.ROOT/f'{fam}_raw.npz');own,bet,size=raw['own'],raw['bet'],raw['size'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];np.testing.assert_array_equal(fv,np.array([tf[t] for t in d['table_id']]));af=fv[g];pp=np.zeros((len(d),2));dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2))
  for f in range(4):
   xx=np.zeros((len(a),15),np.float32)
   for h in range(4):
    if h==f:continue
    mask=(af==f)|(af==h);z=extra(own[mask],bet[mask],size[mask],f,h,pc);ix=np.flatnonzero(mask);xx[ix[af[mask]==h]]=z[af[mask]==h];xx[ix[af[mask]==f]]+=z[af[mask]==f]/3
   np.savez_compressed(ROOT/f'{fam}_extra_fold{f}.npz',x=xx);x=np.column_stack([v['x'],xx])
   if fam=='directed_transfer':y,tr,va=old.labels(d,a,f)
   else:yy,e,_=old.target(d,f);y=yy[g];tr=e[g];va=af==f
   assert not(tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=3,allow_writing_files=False,verbose=False);m.fit(x[tr],y[tr]);m.save_model(str(ROOT/f'{fam}_primary_fold{f}.cbm'));pp[g[va],actor[va]]=m.predict_proba(x[va],thread_count=3)[:,1];audit.append({'family':fam,'fold':f,'training_actions':int(tr.sum()),'positive_actions':int(y[tr].sum()),'validation_overlap':0});print('nested joint',fam,f,round(time.time()-start,1),flush=True)
  parts.append(d.select('pair_id','hand_id').with_columns(pl.Series('replacement',(pp*dw).sum(1))))
 base.join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('replacement','bg_primary').alias('bg_primary')).drop('replacement').write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':main()
