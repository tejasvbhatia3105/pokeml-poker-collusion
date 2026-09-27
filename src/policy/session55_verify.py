import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session55_current_targets import ROOT,OLD,state,direct,soft_support,soft,isolation,C
from session50_matchup import ROOT as PAIR_ROOT
from session6_priority import inclusion
def predict(path,x):
 m=CatBoostClassifier();m.load_model(str(path));return m.predict_proba(x,thread_count=2)
def verify(pilot=False):
 s=state();halfmap=json.load(open(OLD/'table_half_split.json'));jobs=[(0,0,None,True)]+([(0,1,0,False)] if pilot else [(o,p,h,False) for o in range(4) for p in range(4) if p!=o for h in range(2)]);maxerr=0;models=0;mutations=0;audited=0
 for outer,parent,half,control in jobs:
  tag=f'outer{outer}_parent{parent}_'+('control' if control else f'half{half}');root=ROOT/tag;saved=pl.read_parquet(ROOT/(tag+'.parquet'));audit=json.load(open(ROOT/(tag+'.json')))
  for fam,v in s.items():
   d=v['d'];fv=v['fv'];hv=np.array([halfmap[t] for t in d['table_id']]);held=(fv==outer) if control else ((fv==parent)&(hv==half));tr=(fv!=outer)&~held;vi=np.flatnonzero(held);a=v['a'];g=a['row'].to_numpy();ar=a['actor'].to_numpy();entry=next(z for z in audit['records'] if z['family']==fam);assert entry['training_tables']==sorted(set(d['table_id'].to_numpy()[tr]));assert entry['prediction_tables']==sorted(set(d['table_id'].to_numpy()[held]));assert not set(entry['training_tables'])&(set(entry['prediction_tables'])|set(entry['outer_excluded_tables']));dm=d.with_columns(pl.when(pl.Series(~tr)).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when(pl.Series(~tr)).then(999).otherwise(C('evidence_rank')).alias('evidence_rank'),pl.when(pl.Series(~tr)).then(99).otherwise(C('subtype')).alias('subtype'),pl.when(pl.Series(~tr)).then(-100).otherwise(C('time')).alias('time'));p0=np.zeros(len(d));p1=np.full(len(d),np.nan)
   if fam=='soft_play':
    sub,known,_=soft_support(d,tr);ms,mk,_=soft_support(dm,tr);np.testing.assert_array_equal(sub,ms);np.testing.assert_array_equal(known,mk);assert not known[~tr].any();tp=predict(root/'soft_type.cbm',v['xt']);models+=1;y,e,rec=soft(d,tr,tp);my,me,mrec=soft(dm,tr,tp);np.testing.assert_array_equal(y,my);np.testing.assert_array_equal(e,me);assert rec==mrec==entry['assignments'];et=e[g];assert not et[held[g]].any();p0[g[held[g]]]=predict(root/'soft_primary.cbm',v['x'][held[g]])[:,1];models+=1
   else:
    typepath=f'artifacts/evidence_session6/priority_ordered_type_{fam}_fold{parent}.cbm' if control else f'artifacts/evidence_session36_nested_cascade/outer{outer}_parent{parent}_half{half}_{fam}_type.cbm';tp=predict(typepath,v['xt'])[:,1]
    if fam=='directed_transfer':
     y,e,donor,rec=direct(d,tr,tp);my,me,md,mrec=direct(dm,tr,tp)
     for x,z in [(y,my),(e,me),(donor,md)]:np.testing.assert_array_equal(x,z)
     assert rec==mrec==entry['assignments'];AX=np.load('artifacts/evidence_session25_persistent_actor/actor_features.npz')['hand'];PX=np.load('artifacts/evidence_session25_persistent_actor/actor_features.npz')['pair'];groups=[q['row'].to_numpy() for _,q in d.group_by('pair_id',maintain_order=True)];first=np.array([ix[0] for ix in groups]);pt=tr[first]&(donor[first]>=0);pv=held[first];assert not(pt&pv).any();pr=predict(root/'actor.cbm',PX[pv].reshape(-1,PX.shape[-1]))[:,1].reshape(-1,2);models+=1;pr/=pr.sum(1)[:,None];dw=np.zeros((len(d),2))
     for ix,p in zip([groups[i] for i in np.flatnonzero(pv)],pr):dw[ix]=p
     et=e[g,0]&(ar==donor[g]);assert not et[held[g]].any() and not e[held].any();pp=np.zeros((len(d),2));pp[g[held[g]],ar[held[g]]]=predict(root/'direct_primary.cbm',v['x'][held[g]])[:,1];models+=1;p0=(pp*dw).sum(1);xv=np.concatenate([np.column_stack([v['hx'][vi],AX[vi,r]]) for r in range(2)]);ps=predict(root/'direct_secondary.cbm',xv)[:,1].reshape(2,-1).T;models+=1;p1[vi]=(ps*dw[vi]).sum(1)
    else:
     original=isolation(d,tr,tp);mutated=isolation(dm,tr,tp)
     for x,z in zip(original,mutated):np.testing.assert_array_equal(x,z)
     assert not original[2][held].any() and not original[3][held].any();p0[held]=predict(root/'iso_event1.cbm',v['x'][held])[:,1];p1[held]=predict(root/'iso_event2.cbm',v['x'][held])[:,1];models+=2
   ref=d.filter(pl.Series(held)).select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');maxerr=max(maxerr,float(abs(p0[held]-ref['new_primary'].to_numpy()).max()))
   if fam!='soft_play':maxerr=max(maxerr,float(abs(p1[held]-ref['new_secondary'].to_numpy()).max()))
   else:assert ref['new_secondary'].is_nan().all()
   mutations+=1;audited+=len(vi)
  print('nested verify',tag,'models',models,'maxerror',maxerr,flush=True)
 derived_error=0;val_error=0
 if not pilot:
  expected=pl.read_parquet(PAIR_ROOT/'current/event_oof.parquet')
  for outer in range(4):
   d=pl.read_parquet(ROOT/f'nested_outer{outer}.parquet');old=pl.read_parquet(OLD/f'nested_outer{outer}.parquet');q=d.join(old.select('pair_id','hand_id','base','hist_primary','hist_secondary'),on=['pair_id','hand_id'],suffix='_old',validate='1:1');assert all((q[c]==q[c+'_old']).all() for c in ['base','hist_primary','hist_secondary']);val=d.filter(C('fold')==outer).join(expected.select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1');val_error=max(val_error,float((val['cat_primary']-val['bg_primary']).abs().max()),float((val['cat_secondary']-val['bg_secondary']).abs().max()))
   for _,g in d.group_by('pair_id'):
    g=g.sort('time','hand_id');ca=g.select('cat_primary','cat_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];ci=inclusion(*ca.T);ji=inclusion(*((ca+hp)/2).T);derived_error=max(derived_error,float(abs(ci-g['cat_inclusion'].to_numpy()).max()),float(abs(ji-g['joint_inclusion'].to_numpy()).max()),float(abs(.25*g['base'].to_numpy()+.25*ci+.5*ji-g['r29'].to_numpy()).max()))
 assert maxerr==0 and val_error==0 and derived_error==0;report={'pilot_only':pilot,'saved_models_replayed':models,'event_probability_error':maxerr,'target_label_rank_time_mutation_cases':mutations,'prediction_hands_across_blocks':audited,'outer_validation_R32_event_error':val_error,'derived_feature_error':derived_error,'old_base_HGB_unchanged':not pilot,'all_training_prediction_and_outer_pool_sets_disjoint':True};name='pilot_verification.json' if pilot else 'input_verification.json';(ROOT/name).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 import sys
 verify('--pilot' in sys.argv)
