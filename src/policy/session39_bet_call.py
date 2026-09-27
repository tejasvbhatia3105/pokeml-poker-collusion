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
from session37_bet_fold import paired_features,OLD
from session27_donor_call_witness import noisy_or
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session39_bet_call');PREV=Path('artifacts/evidence_session27_donor_calls');BASE=Path('artifacts/evidence_session37_bet_fold/paired_hand')
def targets(d,f):
 by={r['pair_id']:r for r in json.load(open(OLD/f'assignments_fold{f}.json'))};y=np.zeros(len(d),bool);e=np.zeros(len(d),bool);donor=np.full(len(d),-1);tv=d['time'].to_numpy()
 for (pid,),q in d.group_by('pair_id'):
  if pid not in by:continue
  assert q['fold'][0]!=f;ix=q['row'].to_numpy();p=q.filter(C('evidence')==1).sort('evidence_rank')['row'].to_numpy();k=by[pid]['cut'];donor[ix]=by[pid]['donor'];e[ix]=True;y[p[k:]]=True
  if len(p)==5:
   if k==5:e[ix]=False
   else:e[ix[tv[ix]>tv[p[-1]]]]=False
 return y,e,donor
def data():
 d=hand_data().filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');a=pl.read_parquet(PREV/'call_actions.parquet').with_row_index('action_row');cfg=json.load(open(PREV/'columns.json'));return d,a,cfg
def extra_features(d,a):
 extra,align=paired_features(d,a,response_class=2);extra=extra.rename({c:c.replace('fold_minus_bet_','call_minus_bet_') for c in extra.columns});align=align.rename({'fold_action_no':'call_action_no'});return extra,align
def main():
 ROOT.mkdir(exist_ok=True);d,a,cfg=data();extra,align=extra_features(d,a);extra.write_parquet(ROOT/'action_features.parquet');align.write_parquet(ROOT/'alignment.parquet');ec=[c for c in extra.columns if c!='action_row'];g=a['row'].to_numpy();r=a['actor'].to_numpy();fv=d['fold'].to_numpy();bag=2*g+r;cnt=np.bincount(bag,minlength=2*len(d));sup=cnt.reshape(-1,2)>0;x=np.column_stack([a.select(cfg['action']).to_numpy(),d.select(cfg['hand']).to_numpy()[g],extra.select(ec).to_numpy()]);pp=np.zeros((len(d),2));audit=[];start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'action_columns':cfg['action'],'hand_columns':cfg['hand'],'paired_columns':ec,'schedule':'3 EM rounds Cat400 CrossEntropy depth5 lr .035 L2 8 original secondary seed','target':'same session27 donor-call hand labels, support and censoring','uncertainty':'one or more actual calls may constitute the secondary event; independent noisy-OR assumption'},indent=2))
 for f in range(4):
  y,e,donor=targets(d,f);selected=np.flatnonzero(e);ix=selected[sup[selected,donor[selected]]];assert y[ix].sum()==y.sum();tr=e[g]&(r==donor[g]);va=fv[g]==f;assert not (tr&va).any();gt=g[tr];yt=y[gt];nn=np.bincount(gt,minlength=len(d));resp=yt/nn[gt];trace=[]
  for step in range(3):
   m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='CrossEntropy',thread_count=2,random_seed=6312+11*f,verbose=False,allow_writing_files=False);m.fit(x[tr],resp);m.save_model(str(ROOT/f'secondary_fold{f}_em{step}.cbm'));p=m.predict_proba(x[tr],thread_count=2)[:,1];hp=noisy_or(p,gt,len(d));resp=np.where(yt,p/np.maximum(hp[gt],1e-8),0).clip(0,1);v=hp[ix].clip(1e-8,1-1e-8);trace.append(float(-(y[ix]*np.log(v)+(1-y[ix])*np.log1p(-v)).mean()))
  p=m.predict_proba(x[va],thread_count=2)[:,1];pp[fv==f]=noisy_or(p,bag[va],2*len(d)).reshape(-1,2)[fv==f];audit.append({'fold':f,'training_actions':int(tr.sum()),'eligible_hands':len(ix),'positive_hands':int(y[ix].sum()),'overlap':0,'training_bag_nll':trace});print('bet call',f,round(time.time()-start,1),flush=True)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));d.select('pair_id','hand_id').with_columns(pl.Series('actor0_secondary',pp[:,0]),pl.Series('actor1_secondary',pp[:,1])).write_parquet(ROOT/'conditional_secondary.parquet');dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();base=pl.read_parquet(BASE/'event_oof.parquet')
 for kind in ['original_call','paired_call']:
  root=ROOT/kind;root.mkdir(exist_ok=True)
  if kind=='original_call':q=d.select('pair_id','hand_id').join(pl.read_parquet(PREV/'call_mil/event_oof.parquet').select('pair_id','hand_id',C('bg_secondary').alias('new_secondary')),on=['pair_id','hand_id'],validate='1:1')
  else:q=d.select('pair_id','hand_id').with_columns(pl.Series('new_secondary',(pp*dw).sum(1)))
  base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_secondary').write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
