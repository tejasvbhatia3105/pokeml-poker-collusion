\
\
\
\
\
\
\
import os,json,time,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session55_current_targets import ROOT,OLD,FAMILIES,state,direct,soft_support,soft,isolation,C
from session50_matchup import ROOT as PAIR_ROOT
from session6_priority import inclusion
def fit_predict(path,x,y,xv,seed,iterations=400,depth=5,multi=False):
 m=CatBoostClassifier(iterations=iterations,depth=depth,learning_rate=.035 if iterations==400 else .04,l2_leaf_reg=8 if iterations==400 else 10,loss_function='MultiClass' if multi else 'Logloss',thread_count=2,random_seed=seed,verbose=False,allow_writing_files=False);m.fit(x,y);p=m.predict_proba(xv,thread_count=2);m.save_model(str(path));r=CatBoostClassifier();r.load_model(str(path));assert np.array_equal(p,r.predict_proba(xv,thread_count=2));return p
def train_block(s,outer,parent,half,control=False):
 tag=f'outer{outer}_parent{parent}_'+('control' if control else f'half{half}');path=ROOT/(tag+'.parquet')
 if path.exists():return pl.read_parquet(path)
 root=ROOT/tag;root.mkdir(exist_ok=True);halfmap=json.load(open(OLD/'table_half_split.json'));parts=[];audits=[];saved=0
 for fam,v in s.items():
  d=v['d'];fv=v['fv'];hv=np.array([halfmap[t] for t in d['table_id']]);held=(fv==outer) if control else ((fv==parent)&(hv==half));tr=(fv!=outer)&~held;vi=np.flatnonzero(held);a=v['a'];g=a['row'].to_numpy();ar=a['actor'].to_numpy();p0=np.zeros(len(d));p1=np.full(len(d),np.nan);training_tables=sorted(set(d['table_id'].to_numpy()[tr]));prediction_tables=sorted(set(d['table_id'].to_numpy()[held]));assert not set(training_tables)&set(prediction_tables)
  if fam=='soft_play':
   sub,known,pending=soft_support(d,tr);assert set(sub[known])=={0,1,2};tp=fit_predict(root/'soft_type.cbm',v['xt'][known],sub[known],v['xt'],6210+parent,200,3,True);saved+=1;y,e,assignments=soft(d,tr,tp);et=e[g];va=held[g];assert not (et&held[g]).any();assert y.sum()==y[g[et]].sum();p=fit_predict(root/'soft_primary.cbm',v['x'][et],y[g[et]],v['x'][va],6311+11*parent)[:,1];saved+=1;p0[g[va]]=p
  else:
   typepath=f'artifacts/evidence_session6/priority_ordered_type_{fam}_fold{parent}.cbm' if control else f'artifacts/evidence_session36_nested_cascade/outer{outer}_parent{parent}_half{half}_{fam}_type.cbm';typ=CatBoostClassifier();typ.load_model(typepath);tp=typ.predict_proba(v['xt'],thread_count=2)[:,1]
   if fam=='directed_transfer':
    y,e,donor,assignments=direct(d,tr,tp);AX=np.load('artifacts/evidence_session25_persistent_actor/actor_features.npz')['hand'];PX=np.load('artifacts/evidence_session25_persistent_actor/actor_features.npz')['pair'];groups=[q['row'].to_numpy() for _,q in d.group_by('pair_id',maintain_order=True)];first=np.array([ix[0] for ix in groups]);pt=tr[first]&(donor[first]>=0);pv=held[first];yd=np.column_stack([donor[first[pt]]==0,donor[first[pt]]==1]).reshape(-1).astype(int);pr=fit_predict(root/'actor.cbm',PX[pt].reshape(-1,PX.shape[-1]),yd,PX[pv].reshape(-1,PX.shape[-1]),25000+parent,300,3)[:,1].reshape(-1,2);saved+=1;pr/=pr.sum(1)[:,None];dw=np.zeros((len(d),2))
    for ix,p in zip([groups[i] for i in np.flatnonzero(pv)],pr):dw[ix]=p
    et=e[g,0]&(ar==donor[g]);va=held[g];assert y[:,0].sum()==y[g[et],0].sum();pp=np.zeros((len(d),2));pp[g[va],ar[va]]=fit_predict(root/'direct_primary.cbm',v['x'][et],y[g[et],0],v['x'][va],6311+11*parent)[:,1];saved+=1;p0=(pp*dw).sum(1);ti=np.flatnonzero(e[:,1]);xx=np.column_stack([v['hx'][ti],AX[ti,donor[ti]]]);xv=np.concatenate([np.column_stack([v['hx'][vi],AX[vi,r]]) for r in range(2)]);ps=fit_predict(root/'direct_secondary.cbm',xx,y[ti,1],xv,6312+11*parent)[:,1].reshape(2,-1).T;saved+=1;p1[vi]=(ps*dw[vi]).sum(1)
   else:
    y1,y2,e1,e2=isolation(d,tr,tp);assignments=[]
    for k,(y,e) in enumerate([(y1,e1),(y2,e2)]):
     assert not e[held].any();p=fit_predict(root/f'iso_event{k+1}.cbm',v['x'][e],y[e],v['x'][held],6311+11*parent+k)[:,1];saved+=1
     if k==0:p0[held]=p
     else:p1[held]=p
  assert np.isfinite(p0[held]).all();assert fam=='soft_play' or np.isfinite(p1[held]).all();parts.append(d.filter(pl.Series(held)).select('pair_id','hand_id').with_columns(pl.Series('new_primary',p0[held]),pl.Series('new_secondary',p1[held])));audits.append({'family':fam,'training_tables':training_tables,'prediction_tables':prediction_tables,'outer_excluded_tables':sorted(set(d['table_id'].to_numpy()[fv==outer])),'prediction_hands':int(held.sum()),'assignments':assignments})
 out=pl.concat(parts);out.write_parquet(path);path.with_suffix('.json').write_text(json.dumps({'outer':outer,'parent':parent,'half':half,'control':control,'new_models_saved_and_replayed':saved,'records':audits},indent=2));return out
def combine(outer,pieces):
 val=pl.read_parquet(PAIR_ROOT/'current/event_oof.parquet').filter(C('fold')==outer).select('pair_id','hand_id',C('bg_primary').alias('new_primary'),C('bg_secondary').alias('new_secondary'));raw=pl.read_parquet(OLD/f'nested_outer{outer}.parquet').join(pl.concat(pieces+[val]),on=['pair_id','hand_id'],validate='1:1').with_columns(C('new_primary').alias('cat_primary'),pl.when(C('new_secondary').is_nan()).then(C('cat_secondary')).otherwise(C('new_secondary')).alias('cat_secondary')).drop('new_primary','new_secondary');parts=[]
 for _,g in raw.group_by('pair_id'):
  g=g.sort('time','hand_id');ca=g.select('cat_primary','cat_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];ci=inclusion(*ca.T);ji=inclusion(*((ca+hp)/2).T);parts.append(g.with_columns(pl.Series('cat_inclusion',ci),pl.Series('joint_inclusion',ji),pl.Series('r29',.25*g['base'].to_numpy()+.25*ci+.5*ji)))
 out=pl.concat(parts);assert len(out)==45129;out.write_parquet(ROOT/f'nested_outer{outer}.parquet')
def main(pilot=False):
 ROOT.mkdir(exist_ok=True);assert (ROOT/'target_replay.json').exists();assert __import__('pathlib').Path('artifacts/evidence_session36_nested_cascade/input_verification.json').exists();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'split_source':str(OLD/'table_half_split.json'),'target_verification':'all12 outer family/fold target reconstructions match R32 exactly','models_per_inner_block':7,'saved_models':'all retained and immediately reloaded, prediction replay exact','inner_label_exclusion':'outer fold and one parent-half block never enter new fitting/assignment; borrowed subtype models use same exclusion','future_correction':'not yet trained'},indent=2));s=state();start=time.time()
 if pilot:
  out=train_block(s,0,0,None,True);ref=out.join(pl.read_parquet(PAIR_ROOT/'current/event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1');err0=float((ref['new_primary']-ref['bg_primary']).abs().max());err1=float((ref.filter(~C('new_secondary').is_nan())['new_secondary']-ref.filter(~C('new_secondary').is_nan())['bg_secondary']).abs().max());assert max(err0,err1)==0;train_block(s,0,1,0);(ROOT/'pilot.json').write_text(json.dumps({'outer0_control_event_errors':[err0,err1],'inner_block':'outer0_parent1_half0','models_saved_and_replayed':14,'seconds':time.time()-start},indent=2));print('nested pilot complete',err0,err1,round(time.time()-start,1),flush=True);return
 assert (ROOT/'pilot.json').exists()
 for outer in range(4):
  pieces=[]
  for parent in range(4):
   if parent==outer:continue
   for half in range(2):pieces.append(train_block(s,outer,parent,half));print('nested current',outer,parent,half,round(time.time()-start,1),flush=True)
  combine(outer,pieces)
 print('nested current complete',round(time.time()-start,1),flush=True)
if __name__=='__main__':
 import sys
 main('--pilot' in sys.argv)
