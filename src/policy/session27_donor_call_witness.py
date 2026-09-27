import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session26_exact_fold_witness import actions
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session27_donor_calls');OLD=Path('artifacts/evidence_session25_persistent_actor')
def noisy_or(p,g,n):return -np.expm1(np.bincount(g,weights=np.log1p(-p.clip(1e-8,1-1e-8)),minlength=n))
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'primary':'unchanged session26 action+hand fold witness head','call_hand':'original Cat400 on 811 hand +170 donor-oriented attributes, eligible donor-call hands only','call_mil':'3 EM rounds Cat400 CrossEntropy; exact independent noisy-OR posterior p(action)/p(any) in positive eligible bags; original hand + raw call state inputs','supervision':'session25 persistent actor and cut assignments only from outer-training labels','inference':'no call => component probability zero; latent donor weighted by saved heldout gameplay classifier','caveat':'inferred action support and tier labels; no claim all calls in a true hand are true evidence actions'},indent=2));full=hand_data();d=full.filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');a,ac=actions(d,2);a.write_parquet(ROOT/'call_actions.parquet');cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];hx=d.select(cols).to_numpy();AX=np.load(OLD/'actor_features.npz')['hand'];x=np.column_stack([a.select(ac).to_numpy(),hx[a['row'].to_numpy()]]);g=a['row'].to_numpy();r=a['actor'].to_numpy();fv=d['fold'].to_numpy();bag=2*g+r;cnt=np.bincount(bag,minlength=2*len(d));sup=cnt.reshape(-1,2)>0;pred={k:np.zeros((len(d),2)) for k in ['call_hand','call_mil']};audit=[];start=time.time();(ROOT/'columns.json').write_text(json.dumps({'action':ac,'hand':cols},indent=2))
 for f in range(4):
  by={r['pair_id']:r for r in json.load(open(OLD/f'assignments_fold{f}.json'))};y=np.zeros(len(d),bool);e=np.zeros(len(d),bool);donor=np.full(len(d),-1)
  for (pid,),q in d.group_by('pair_id'):
   if pid not in by:continue
   assert q['fold'][0]!=f;ix=q['row'].to_numpy();p=q.filter(C('evidence')==1).sort('evidence_rank')['row'].to_numpy();k=by[pid]['cut'];donor[ix]=by[pid]['donor'];e[ix]=True;y[p[k:]]=True
   if len(p)==5:
    if k==5:e[ix]=False
    else:e[ix[d['time'].to_numpy()[ix]>d['time'].to_numpy()[p[-1]]]]=False
  selected=np.flatnonzero(e);support=sup[selected,donor[selected]];ix=selected[support];assert int(y[ix].sum())==int(y.sum());vi=np.flatnonzero(fv==f);tx=np.column_stack([hx[ix],AX[ix,donor[ix]]]);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6312+11*f,verbose=False,allow_writing_files=False);root=ROOT/'call_hand';root.mkdir(exist_ok=True);m.fit(tx,y[ix]);m.save_model(str(root/f'secondary_fold{f}.cbm'))
  for j in range(2):
   jj=vi[sup[vi,j]];pred['call_hand'][jj,j]=m.predict_proba(np.column_stack([hx[jj],AX[jj,j]]),thread_count=2)[:,1]
  tr=e[g]&(r==donor[g]);va=fv[g]==f;assert not (tr&va).any();gt=g[tr];yt=y[gt];nn=np.bincount(gt,minlength=len(d));resp=yt/nn[gt];trace=[];root=ROOT/'call_mil';root.mkdir(exist_ok=True)
  for step in range(3):
   m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='CrossEntropy',thread_count=2,random_seed=6312+11*f,verbose=False,allow_writing_files=False);m.fit(x[tr],resp);m.save_model(str(root/f'secondary_fold{f}_em{step}.cbm'));p=m.predict_proba(x[tr],thread_count=2)[:,1];hp=noisy_or(p,gt,len(d));resp=np.where(yt,p/np.maximum(hp[gt],1e-8),0).clip(0,1);v=hp[ix].clip(1e-8,1-1e-8);trace.append(float(-(y[ix]*np.log(v)+(1-y[ix])*np.log1p(-v)).mean()))
  p=m.predict_proba(x[va],thread_count=2)[:,1];out=noisy_or(p,bag[va],2*len(d)).reshape(-1,2);pred['call_mil'][vi]=out[vi];audit.append({'fold':f,'eligible_call_hands':len(ix),'positive_call_hands':int(y[ix].sum()),'unique_call_positive_hands':int(((nn==1)&y).sum()),'positive_call_actions':int(yt.sum()),'training_call_actions':int(tr.sum()),'validation_overlap':0,'mil_training_bag_nll':trace});print('donor calls',f,round(time.time()-start,1),flush=True)
 base=pl.read_parquet('artifacts/evidence_session26_exact_fold/action_hand/event_oof.parquet');dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
 for kind,pp in pred.items():
  root=ROOT/kind;d.select('pair_id','hand_id').with_columns(pl.Series('actor0_secondary',pp[:,0]),pl.Series('actor1_secondary',pp[:,1])).write_parquet(root/'conditional_secondary.parquet');q=d.select('pair_id','hand_id').with_columns(pl.Series('new_secondary',(pp*dw).sum(1)));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_secondary').write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
