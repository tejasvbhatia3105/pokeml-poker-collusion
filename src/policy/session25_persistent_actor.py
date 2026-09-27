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
from session10_roles import ATTR,pf
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session25_persistent_actor')

def actor_features(d,players=None):
 labs=(pl.read_csv('data/development_labels.csv') if players is None else players).select('pair_id','player_1','player_2');parts=[]
 for (table,),g in d.group_by('table_id'):
  s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(g.select('hand_id').unique(),on='hand_id',how='semi');s=s.with_columns(pl.Series('pf',pf(s['hole_card_1'],s['hole_card_2'])))
  q=g.select('pair_id','hand_id').join(labs,on='pair_id').join(s.select('hand_id',C('player_id').alias('player_1'),C('pf').alias('pf1')),on=['hand_id','player_1'],validate='m:1').join(s.select('hand_id',C('player_id').alias('player_2'),C('pf').alias('pf2')),on=['hand_id','player_2'],validate='m:1')
  parts.append(q.select('pair_id','hand_id',(C('pf1')<=C('pf2')).alias('weak_first')))
 q=d.select('pair_id','hand_id').join(pl.concat(parts),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');w=q['weak_first'].to_numpy();a=d.select(['weak_'+c for c in ATTR]).to_numpy();b=d.select(['strong_'+c for c in ATTR]).to_numpy();own=np.where(w[:,None],a,b);opp=np.where(w[:,None],b,a)
 return np.stack([np.column_stack([own,opp]),np.column_stack([opp,own])],1)

def pair_features(x,groups):
 rows=[]
 for ix in groups:
  z=x[ix];rows.append(np.concatenate([z.mean(0),z.std(0),z.min(0),z.max(0)],1))
 return np.stack(rows)

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'actor_model':'Cat 300 depth3 lr .04 L2 10; mirrored per-pair moments of 85 own/85 other attributes','event_model':'original Cat400 depth5 schedule; known training donor orientation; donor inferred on heldout gameplay','controls':'same grounded consistent-actor labels with original 811 invariant inputs','caveat':'hypothesis explored on reused public truth; exceptions excluded only from training; actor IDs only join keys'},indent=2))
 full=hand_data();d=full.filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];pids=[d['pair_id'][int(ix[0])] for ix in groups];pfold=np.array([d['fold'][int(ix[0])] for ix in groups]);fv=d['fold'].to_numpy();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));X=d.select(cols['event']).to_numpy();XT=d.select(cols['type']).to_numpy();AX=actor_features(d);PX=pair_features(AX,groups);np.savez_compressed(ROOT/'actor_features.npz',hand=AX,pair=PX);truth={r['pair_id']:r for r in json.load(open('artifacts/evidence_session25_actor_consistency/audit.json'))['records']};donor=np.array([next(iter({a for o in truth[p]['options'] for a in o['consistent_actors']}),-1) for p in pids]);pred={k:np.zeros((len(d),2)) for k in ['control','oriented']};actor_oof=np.zeros((len(groups),2));conditional=np.zeros((len(d),2,2));audit=[];start=time.time()
 for f in range(4):
  tr=(pfold!=f)&(donor>=0);va=pfold==f;pi=np.flatnonzero(tr);yv=np.column_stack([donor[tr]==0,donor[tr]==1]).reshape(-1).astype(int);dm=CatBoostClassifier(iterations=300,depth=3,learning_rate=.04,l2_leaf_reg=10,thread_count=2,random_seed=25000+f,verbose=False,allow_writing_files=False);dm.fit(PX[tr].reshape(-1,PX.shape[-1]),yv);dm.save_model(str(ROOT/f'actor_fold{f}.cbm'));prob=dm.predict_proba(PX[va].reshape(-1,PX.shape[-1]),thread_count=2)[:,1].reshape(-1,2);prob/=prob.sum(1)[:,None];actor_oof[va]=prob
                                                                                
                                                                           
  typ=CatBoostClassifier();typ.load_model(f'artifacts/evidence_session6/priority_ordered_type_directed_transfer_fold{f}.cbm');tp=typ.predict_proba(XT,thread_count=2)[:,1];y=np.zeros((len(d),2),bool);eligible=np.zeros_like(y);selected=np.zeros_like(AX[:,0]);records=[]
  for i in pi:
   ix=groups[i];g=d[ix];pos=g.filter(C('evidence')==1).sort('evidence_rank')['row'].to_numpy();opts=[o for o in truth[pids[i]]['options'] if int(donor[i]) in o['consistent_actors']];ll=[]
   for o in opts:
    k=o['cut'];ll.append(float(np.log(tp[pos[:k]].clip(1e-8,1)).sum()+np.log((1-tp[pos[k:]]).clip(1e-8,1)).sum()))
   k=opts[int(np.argmax(ll))]['cut'];yy=np.r_[np.zeros(k,int),np.ones(len(pos)-k,int)];y[pos,yy]=True;eligible[ix]=True;selected[ix]=AX[ix,donor[i]]
   if len(pos)==5:
    final=yy[-1];eligible[ix,final+1:]=False;after=ix[d['time'].to_numpy()[ix]>d['time'].to_numpy()[pos[-1]]];eligible[after,final]=False
   records.append({'pair_id':pids[i],'donor':int(donor[i]),'cut':k})
  assert not eligible[fv==f].any();(ROOT/f'assignments_fold{f}.json').write_text(json.dumps(records,indent=2));vi=np.flatnonzero(fv==f);dw=np.zeros((len(d),2))
  for i,p in zip(np.flatnonzero(va),prob):dw[groups[i]]=p
  for kind in pred:
   root=ROOT/kind;root.mkdir(exist_ok=True)
   for k in range(2):
    e=eligible[:,k];xx=X if kind=='control' else np.column_stack([X,selected]);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k+1,verbose=False,allow_writing_files=False);m.fit(xx[e],y[e,k]);m.save_model(str(root/f'event{k+1}_directed_transfer_fold{f}.cbm'))
    if kind=='control':pred[kind][vi,k]=m.predict_proba(X[vi],thread_count=2)[:,1]
    else:
     pp=np.column_stack([m.predict_proba(np.column_stack([X[vi],AX[vi,r]]),thread_count=2)[:,1] for r in range(2)]);conditional[vi,:,k]=pp;pred[kind][vi,k]=(pp*dw[vi]).sum(1)
    audit.append({'fold':f,'arm':kind,'head':k+1,'eligible_rows':int(e.sum()),'positive_rows':int(y[e,k].sum()),'actor_train_pairs':int(tr.sum()),'validation_overlap':int((e&(fv==f)).sum())})
  print('persistent actor',f,round(time.time()-start,1),flush=True)
 pl.DataFrame({'pair_id':pids,'fold':pfold,'audit_truth_actor':donor,'actor0':actor_oof[:,0],'actor1':actor_oof[:,1]}).write_parquet(ROOT/'actor_oof.parquet');d.select('pair_id','hand_id').with_columns(*[pl.Series(f'actor{r}_event{k}',conditional[:,r,k]) for r in range(2) for k in range(2)]).write_parquet(ROOT/'conditional_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
 original=pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet').select('pair_id','hand_id',C('primary').alias('bg_primary'),C('secondary').alias('bg_secondary'))
 for kind,pp in pred.items():
  q=d.select('pair_id','hand_id').with_columns(pl.Series('new1',pp[:,0]),pl.Series('new2',pp[:,1]));out=full.select('pair_id','hand_id','fold','time','behavior_family').join(original,on=['pair_id','hand_id'],validate='1:1').join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new1','bg_primary').alias('bg_primary'),pl.coalesce('new2','bg_secondary').alias('bg_secondary')).drop('new1','new2');out.write_parquet(ROOT/kind/'event_oof.parquet');assemble(ROOT/kind)
 known=donor>=0;print('actor accuracy',float((actor_oof.argmax(1)[known]==donor[known]).mean()),'logloss',float(-np.log(actor_oof[np.flatnonzero(known),donor[known]].clip(1e-9,1)).mean()),flush=True)
if __name__=='__main__':main()
