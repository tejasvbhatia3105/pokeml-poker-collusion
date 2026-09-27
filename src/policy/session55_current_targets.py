\
\
\
\
\
\
import json
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session50_matchup import load,ROOT as PAIR_ROOT
from session8_data import hand_data,targets as original_targets
from session17_grounded_tiers import assignments
from session6_priority import training_targets
from session35_fold_likelihood import labels as fold_labels
from session39_bet_call import targets as secondary_targets
from session29_shared_fold_witness import target as soft_target
C=pl.col;ROOT=Path('artifacts/evidence_session55_current_nested');OLD=Path('artifacts/evidence_session10/nested6')
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
def state():
 cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));out={}
 for fam in FAMILIES:
  d,a,x,cols=load(fam);cur=np.load(PAIR_ROOT/fam/'features.npz')['x'][:,[3,6]]
  if fam=='coordinated_isolation':extra=np.full((len(d),2),-2.);extra[a['row'].to_numpy()]=cur
  else:extra=cur
  out[fam]={'d':d,'a':a,'x':np.column_stack([x,extra]),'xt':d.select(cfg['type']).to_numpy(),'hx':d.select(cfg['event']).to_numpy(),'fv':d['fold'].to_numpy()}
 return out
def direct(d,tr,tp):
 truth={r['pair_id']:r for r in json.load(open('artifacts/evidence_session25_actor_consistency/audit.json'))['records']};y=np.zeros((len(d),2),bool);e=np.zeros_like(y);donor=np.full(len(d),-1);records=[];tv=d['time'].to_numpy()
 for (pid,),g in d.group_by('pair_id',maintain_order=True):
  ix=g['row'].to_numpy()
  if not tr[ix[0]]:continue
  assert tr[ix].all();actors={a for o in truth[pid]['options'] for a in o['consistent_actors']};assert len(actors)<=1
  if not actors:continue
  actor=next(iter(actors));pos=g.filter(C('evidence')==1).sort('evidence_rank')['row'].to_numpy();opts=[o for o in truth[pid]['options'] if actor in o['consistent_actors']];ll=[float(np.log(tp[pos[:o['cut']]].clip(1e-8,1)).sum()+np.log((1-tp[pos[o['cut']:]]).clip(1e-8,1)).sum()) for o in opts];k=opts[int(np.argmax(ll))]['cut'];chosen=np.r_[np.zeros(k,int),np.ones(len(pos)-k,int)];y[pos,chosen]=True;e[ix]=True;donor[ix]=actor
  if len(pos)==5:
   final=chosen[-1];e[np.ix_(ix,np.arange(final+1,2))]=False;e[ix[tv[ix]>tv[pos[-1]]],final]=False
  records.append({'pair_id':pid,'donor':actor,'cut':k})
 assert not e[~tr].any();return y,e,donor,records
def soft_support(d,tr):
 sub=np.full(len(d),-1);pending=[]
 for _,g in d.with_columns(C('row').alias('local_row')).group_by('pair_id',maintain_order=True):
  if not tr[int(g['row'][0])]:continue
  ix,valid=assignments(g);assert valid;pending.append((g,ix,valid))
  if len(valid)==1:sub[ix]=valid[0]
 known=(sub>=0)&tr;return sub,known,pending
def soft(d,tr,tp):
 _,_,pending=soft_support(d,tr);y=np.zeros(len(d),bool);e=tr.copy();records=[];tv=d['time'].to_numpy()
 for g,ix,valid in pending:
  ll=[np.log(tp[ix,c].clip(1e-8,1)).sum() for c in valid];chosen=valid[int(np.argmax(ll))];y[ix[chosen==0]]=True
  if len(ix)==5 and np.all(chosen==0):rows=g['row'].to_numpy();e[rows[tv[rows]>tv[ix[-1]]]]=False
  records.append({'pair_id':g['pair_id'][0],'chosen':chosen.tolist()})
 assert not e[~tr].any();return y,e,records
def isolation(d,tr,tp):
 groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];return training_targets(d['evidence'].to_numpy(),d['subtype'].to_numpy(),tp,tr,d['time'].to_numpy(),groups,d['evidence_rank'].fill_null(0).to_numpy())
def replay():
 ROOT.mkdir(exist_ok=True);s=state();full=hand_data();records=[]
 for f in range(4):
  for fam,v in s.items():
   d=v['d'];tr=v['fv']!=f;m=CatBoostClassifier();path=f'artifacts/evidence_session17_grounded/type_fold{f}.cbm' if fam=='soft_play' else f'artifacts/evidence_session6/priority_ordered_type_{fam}_fold{f}.cbm';m.load_model(path);tp=m.predict_proba(v['xt'],thread_count=2)
   if fam=='directed_transfer':
    y,e,donor,r=direct(d,tr,tp[:,1]);a=v['a'];g=a['row'].to_numpy();ar=a['actor'].to_numpy();oldy,oldtr,_=fold_labels(d,a,f);np.testing.assert_array_equal(y[g,0],oldy);np.testing.assert_array_equal(e[g,0]&(ar==donor[g]),oldtr);sy,se,sd=secondary_targets(d,f);np.testing.assert_array_equal(y[:,1],sy);np.testing.assert_array_equal(e[:,1],se);np.testing.assert_array_equal(donor,sd)
   elif fam=='soft_play':
    y,e,r=soft(d,tr,tp);oy,oe,_=soft_target(d,f);np.testing.assert_array_equal(y,oy);np.testing.assert_array_equal(e,oe)
   else:
    new=isolation(d,tr,tp[:,1]);old=original_targets(full,f,fam);mask=full['behavior_family'].to_numpy()==fam
    for a,b in zip(new,old[:4]):np.testing.assert_array_equal(a,b[mask])
   records.append({'fold':f,'family':fam,'targets_match_R32_exact':True})
 (ROOT/'target_replay.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':replay()
