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
from session26_exact_fold_witness import actions
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session29_shared_folds')
def target(d,f):
 da={r['pair_id']:r for r in json.load(open(f'artifacts/evidence_session25_persistent_actor/assignments_fold{f}.json'))};sa={r['pair_id']:r for r in json.load(open(f'artifacts/evidence_session17_grounded/assignments_fold{f}.json'))};y=np.zeros(len(d),bool);e=np.zeros(len(d),bool);donor=np.full(len(d),-1)
 for (pid,),q in d.group_by('pair_id'):
  if q['fold'][0]==f:continue
  ix=q['row'].to_numpy();pos=q.filter(C('evidence')==1).sort('evidence_rank')['row'].to_numpy()
  if q['behavior_family'][0]=='directed_transfer':
   if pid not in da:continue
   k=da[pid]['cut'];donor[ix]=da[pid]['donor'];chosen=np.r_[np.zeros(k,int),np.ones(len(pos)-k,int)]
  else:chosen=np.array(sa[pid]['chosen']);assert len(chosen)==len(pos)
  y[pos[chosen==0]]=True;e[ix]=True
  if len(pos)==5 and np.all(chosen==0):e[ix[d['time'].to_numpy()[ix]>d['time'].to_numpy()[pos[-1]]]]=False
 assert not e[d['fold'].to_numpy()==f].any()
 return y,e,donor
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'models':'Cat400 depth5 lr .035 L2 8 original fold seed','inputs':'raw fold action state +811 original hand features; pooled arm adds family indicator','baseline':'session26 action+hand directed fold head, session25 directed secondary, original Cat for other families','selection':'post-result hypothesis after persistent-actor failed for soft play'},indent=2));full=hand_data();d=full.filter(C('behavior_family').is_in(['directed_transfer','soft_play'])).drop('row').with_row_index('row');a,ac=actions(d);a.write_parquet(ROOT/'fold_actions.parquet');g=a['row'].to_numpy();r=a['actor'].to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();isdir=fam=='directed_transfer';assert a.filter(C('pair_id').is_in(d.filter(C('behavior_family')=='soft_play')['pair_id'].unique())).select('pair_id','hand_id').n_unique()==len(a.filter(C('pair_id').is_in(d.filter(C('behavior_family')=='soft_play')['pair_id'].unique())));cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];hx=d.select(cols).to_numpy();x=np.column_stack([a.select(ac).to_numpy(),hx[g]]);pred={k:np.zeros((len(d),2)) for k in ['soft_only','pooled']};(ROOT/'columns.json').write_text(json.dumps({'action':ac,'hand':cols,'pooled_extra':'is_directed_transfer'},indent=2));audit=[];start=time.time()
 for f in range(4):
  y,e,donor=target(d,f);eligible=e[g]&(~isdir[g]|(r==donor[g]));assert y.sum()==y[g[eligible]].sum()
  for kind in pred:
   tr=eligible.copy();va=fv[g]==f
   if kind=='soft_only':tr&=~isdir[g];va&=~isdir[g]
   xx=x if kind=='soft_only' else np.column_stack([x,isdir[g]]);assert not (tr&va).any();root=ROOT/kind;root.mkdir(exist_ok=True);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6311+11*f,verbose=False,allow_writing_files=False);m.fit(xx[tr],y[g[tr]]);m.save_model(str(root/f'primary_fold{f}.cbm'));pred[kind][g[va],r[va]]=m.predict_proba(xx[va],thread_count=2)[:,1];audit.append({'fold':f,'arm':kind,'eligible_actions':int(tr.sum()),'positive_actions':int(y[g[tr]].sum()),'validation_overlap':0})
  print('shared fold',f,round(time.time()-start,1),flush=True)
 base=pl.read_parquet('artifacts/evidence_session26_exact_fold/action_hand/event_oof.parquet');dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',how='left',validate='m:1',maintain_order='left').select(C('actor0').fill_null(1),C('actor1').fill_null(1)).to_numpy();(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
 for kind,pp in pred.items():
  root=ROOT/kind;q=d.select('pair_id','hand_id','behavior_family').with_columns(pl.Series('new_primary',(pp*dw).sum(1)));q=q.filter(C('behavior_family')=='soft_play') if kind=='soft_only' else q;q=q.drop('behavior_family');d.select('pair_id','hand_id').with_columns(pl.Series('actor0_primary',pp[:,0]),pl.Series('actor1_primary',pp[:,1])).write_parquet(root/'conditional_primary.parquet');base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary')).drop('new_primary').write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
