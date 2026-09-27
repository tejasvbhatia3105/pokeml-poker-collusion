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
from session55_current_targets import state,soft_support
from session25_persistent_actor import actor_features
from session62_grounded_list_boost import grounded
from session8_data import hand_data
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session79_uncertain_targets');C=pl.col
def targets(d,tr,tp,family,hard=False):
 y=np.zeros((len(d),2 if family=='directed_transfer' else 1));e=np.zeros_like(y);donor=np.full(len(d),-1);records=[];timev=d['time'].to_numpy()
 if family=='directed_transfer':
  truth={r['pair_id']:r for r in json.load(open('artifacts/evidence_session25_actor_consistency/audit.json'))['records']};pending=[]
  for (pid,),g in d.group_by('pair_id'):
   rows=g['row'].to_numpy()
   if not tr[rows[0]]:continue
   actors={a for o in truth[pid]['options'] for a in o['consistent_actors']}
   if len(actors)!=1:continue
   actor=next(iter(actors));donor[rows]=actor;pos=g.filter(C('evidence')==1).sort('evidence_rank')['row'].to_numpy();cuts=[o['cut'] for o in truth[pid]['options'] if actor in o['consistent_actors']];options=[np.r_[np.zeros(k,int),np.ones(len(pos)-k,int)] for k in cuts];pending.append((g,pos,options))
 else:_,_,pending=soft_support(d,tr)
 for g,pos,options in pending:
  rows=g['row'].to_numpy();probs=np.column_stack([tp[:,1],tp[:,0]]) if family=='directed_transfer' else tp;ll=np.array([np.log(probs[pos,c].clip(1e-8,1)).sum() for c in options]);weights=np.exp(ll-ll.max());weights/=weights.sum()
  if hard:weights=np.eye(len(weights))[np.argmax(weights)]
  for c,w in zip(options,weights):
   yy=np.zeros((len(rows),y.shape[1]));ee=np.ones_like(yy);lookup={int(r):i for i,r in enumerate(rows)};pp=np.array([lookup[int(r)] for r in pos]);after=timev[rows]>timev[pos[-1]]
   if family=='directed_transfer':
    yy[pp,c]=1
    if len(pos)==5:ee[:,int(c[-1])+1:]=0;ee[after,int(c[-1])]=0
   else:
    yy[pp[c==0],0]=1
    if len(pos)==5 and np.all(c==0):ee[after,0]=0
   assert np.all(yy<=ee);y[rows]+=w*yy;e[rows]+=w*ee
  records.append({'pair_id':g['pair_id'][0],'options':len(options),'entropy':float(-np.sum(weights*np.log(weights.clip(1e-15))))})
 assert not y[~tr].any() and not e[~tr].any();return y,e,donor,records
def main():
 ROOT.mkdir(exist_ok=True);states=state();full=hand_data();gx,gcols=grounded(full);cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');allparts={k:[] for k in ['hard','marginal']};audit=[];start=time.time()
 for family in ['directed_transfer','soft_play']:
  v=states[family];d=v['d'];a=v['a'];g=a['row'].to_numpy();ar=a['actor'].to_numpy();fv=d['fold'].to_numpy();x=v['x'];xt=np.column_stack([v['xt'],gx[full['behavior_family'].to_numpy()==family]]);direct=family=='directed_transfer';ax=actor_features(d) if direct else None;dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if direct else np.ones((len(d),2));pp={k:np.zeros((len(d),2)) for k in allparts}
  for f in range(4):
   typ=CatBoostClassifier();typ.load_model(f'artifacts/evidence_session78_grounded_types/{family}_augmented_fold{f}.cbm');tp=typ.predict_proba(xt,thread_count=2)
   for kind in allparts:
    y,e,donor,records=targets(d,fv!=f,tp,family,hard=kind=='hard');root=ROOT/kind;root.mkdir(exist_ok=True)
    for head in range(2 if direct else 1):
     if head==0:weight=e[g,0]*(ar==donor[g] if direct else 1);target=np.divide(y[g,0],e[g,0],out=np.zeros(len(g)),where=e[g,0]>0);X=x;va=fv[g]==f
     else:
      selected=ax[np.arange(len(d)),np.maximum(donor,0)];weight=e[:,1];target=np.divide(y[:,1],e[:,1],out=np.zeros(len(d)),where=e[:,1]>0);X=np.column_stack([v['hx'],selected]);va=fv==f
     tr=weight>1e-12;assert not (tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='CrossEntropy',random_seed=6311+11*f+head,thread_count=2,verbose=False,allow_writing_files=False);m.fit(X[tr],target[tr],sample_weight=weight[tr]);m.save_model(str(root/f'{family}_head{head+1}_fold{f}.cbm'))
     if head==0:
      conditional=np.zeros((len(d),2));conditional[g[va],ar[va]]=m.predict_proba(X[va],thread_count=2)[:,1];pp[kind][fv==f,0]=(conditional*dw).sum(1)[fv==f]
     else:
      pr=np.column_stack([m.predict_proba(np.column_stack([v['hx'][va],ax[va,r]]),thread_count=2)[:,1] for r in [0,1]]);pp[kind][va,1]=(pr*dw[va]).sum(1)
     audit.append({'family':family,'fold':f,'arm':kind,'head':head+1,'training_rows':int(tr.sum()),'expected_positive_mass':float(np.sum(target[tr]*weight[tr])),'eligible_mass':float(weight.sum()),'fractional_targets':int(((target>0)&(target<1)&tr).sum()),'mean_assignment_entropy':float(np.mean([r['entropy'] for r in records])),'validation_overlap':0})
   print('uncertain labels',family,f,round(time.time()-start,1),flush=True)
  for kind in allparts:allparts[kind].append(d.select('pair_id','hand_id').with_columns(pl.Series('replacement_primary',pp[kind][:,0]),pl.Series('replacement_secondary',pp[kind][:,1] if direct else np.full(len(d),np.nan))))
 for kind,parts in allparts.items():
  root=ROOT/kind;z=base.join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('replacement_primary','bg_primary').alias('bg_primary'),pl.when(C('replacement_secondary').is_not_null()&~C('replacement_secondary').is_nan()).then(C('replacement_secondary')).otherwise(C('bg_secondary')).alias('bg_secondary')).drop('replacement_primary','replacement_secondary');z.write_parquet(root/'event_oof.parquet');assemble(root)
 (ROOT/'audit.json').write_text(json.dumps({'method':__doc__,'fits':audit},indent=2))
if __name__=='__main__':main()
