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
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session26_exact_fold');OLD=Path('artifacts/evidence_session25_persistent_actor')
def actions(d,action_class=0):
 a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet').filter((C('action_class')==action_class)&C('facing_partner'));ac=[c for c in a.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family','time']];a=a.select('pair_id','hand_id',*ac);labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');parts=[]
 for (table,),q in d.group_by('table_id'):
  z=a.join(q.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='m:1');raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').select('hand_id','street_no','action_no','player_id');z=z.join(raw,on=['hand_id','street_no','action_no'],validate='m:1').join(labs,on='pair_id',validate='m:1');z=z.with_columns((C('player_id')==C('player_2')).cast(pl.Int8).alias('actor'));assert z.select(((C('player_id')==C('player_1'))|(C('player_id')==C('player_2'))).all()).item();parts.append(z.select('pair_id','hand_id','row','actor',*ac))
 out=pl.concat(parts)
 if action_class==0:assert out.select('pair_id','hand_id','actor').n_unique()==len(out)
 assert out.select('pair_id','hand_id','actor','action_no').n_unique()==len(out)
 return out,ac
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'event_head':'Cat400 depth5 lr .035 L2 8 original seed','arms':['action','action_hand'],'new_labels':'only the primary fold tier; persistent actor and cut from session25 outer-training assignments','limitation':'conditional hard-support hypothesis, two known exceptions remain in validation, no new assertion of action-level ground truth'},indent=2));full=hand_data();d=full.filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');a,ac=actions(d);a.write_parquet(ROOT/'fold_actions.parquet');cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];(ROOT/'columns.json').write_text(json.dumps({'action':ac,'hand':cols},indent=2));hx=d.select(cols).to_numpy();ax=a.select(ac).to_numpy();g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=d['fold'].to_numpy();conditional=d.select('pair_id','hand_id').join(pl.read_parquet(OLD/'conditional_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');prior=conditional.select('actor0_event1','actor1_event1').to_numpy();dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();pred={k:np.zeros((len(d),2)) for k in ['action','action_hand']};audit=[];start=time.time()
 for f in range(4):
  assignments=json.load(open(OLD/f'assignments_fold{f}.json'));by={r['pair_id']:r for r in assignments};y=np.zeros(len(d),bool);e=np.zeros(len(d),bool);donor=np.full(len(d),-1)
  for (pid,),q in d.group_by('pair_id'):
   if pid not in by:continue
   assert q['fold'][0]!=f;ix=q['row'].to_numpy();p=q.filter(C('evidence')==1).sort('evidence_rank')['row'].to_numpy();k=by[pid]['cut'];donor[ix]=by[pid]['donor'];e[ix]=True;y[p[:k]]=True
   if k==5:e[ix[d['time'].to_numpy()[ix]>d['time'].to_numpy()[p[-1]]]]=False
  tr=e[g]&(actor==donor[g]);va=fv[g]==f;assert not (tr&va).any();assert y.sum()==y[g[tr]].sum();audit.append({'fold':f,'training_fold_actions':int(tr.sum()),'unique_positive_fold_actions':int(y[g[tr]].sum()),'heldout_actions':int(va.sum()),'validation_overlap':0})
  for kind in pred:
   root=ROOT/kind;root.mkdir(exist_ok=True);x=ax if kind=='action' else np.column_stack([ax,hx[g]]);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6311+11*f,verbose=False,allow_writing_files=False);m.fit(x[tr],y[g[tr]]);m.save_model(str(root/f'primary_fold{f}.cbm'));pred[kind][g[va],actor[va]]=m.predict_proba(x[va],thread_count=2)[:,1]
  print('exact fold',f,round(time.time()-start,1),flush=True)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));base=pl.read_parquet(OLD/'oriented/event_oof.parquet')
 for kind,pp in pred.items():
  root=ROOT/kind;d.select('pair_id','hand_id').with_columns(pl.Series('actor0_primary',pp[:,0]),pl.Series('actor1_primary',pp[:,1])).write_parquet(root/'conditional_primary.parquet');q=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',(pp*dw).sum(1)));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary')).drop('new_primary').write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
