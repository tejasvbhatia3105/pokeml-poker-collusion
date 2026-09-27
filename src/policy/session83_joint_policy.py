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
from catboost import CatBoostClassifier,CatBoostRegressor
from session55_current_targets import state
from session35_fold_likelihood import labels
from session29_shared_fold_witness import target
from session8_data import hand_data
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session83_joint_policy');C=pl.col
def design(v):
 d,a=v['d'],v['a'];pc=json.load(open('artifacts/policy/feature_columns.json'));parts=[]
 for (table,),q in d.group_by('table_id'):
  local=a.select('action_row','pair_id','hand_id',C('action_no').cast(pl.Int64)).join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').select('hand_id',*pc).with_columns(C('action_no').cast(pl.Int64));parts.append(local.join(raw,on=['hand_id','action_no'],validate='m:1'))
 px=pl.concat(parts).sort('action_row');np.testing.assert_array_equal(px['action_row'].to_numpy(),np.arange(len(a)));own=px.select(pc).to_numpy();basecols=v['x'].shape[1]-2;cfg=json.load(open('artifacts/evidence_session37_bet_fold/config.json'));ec=cfg['paired_columns'];start=len(cfg['fold_columns'])+len(cfg['hand_columns']);bet=v['x'][:,[start+ec.index('bet_'+c) for c in pc]];size=v['x'][:,start+ec.index('bet_log_bet_ratio')];return own,bet,size,pc
def policy_fields(own,bet,size,f):
 m=CatBoostClassifier();m.load_model(f'artifacts/policy/action_fold{f}.cbm');p=m.predict_proba(own,thread_count=2);q=m.predict_proba(bet,thread_count=2);pc=json.load(open('artifacts/policy/feature_columns.json'));callidx=pc.index('call_bb')
 for z,X in [(p,own),(q,bet)]:
  call=X[:,callidx]>0;z[call,1]=0;z[~call,0]=0;z[~call,2]=0;z/=z.sum(1)[:,None]
 sz=CatBoostRegressor();sz.load_model(f'artifacts/policy/size_fold{f}.cbm');res=size-sz.predict(bet,thread_count=2);u=-np.log(p[:,0].clip(1e-7));v=-np.log(q[:,3].clip(1e-7));return np.column_stack([p,q,u,v,u+v,np.minimum(u,v),u-v,res,abs(res)]).astype(np.float32)
def main():
 ROOT.mkdir(exist_ok=True);full=hand_data();states=state();base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');audit=[];parts=[];start=time.time()
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];own,bet,size,pc=design(v);np.savez_compressed(ROOT/f'{fam}_raw.npz',own=own,bet=bet,size=size);g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=d['fold'].to_numpy();dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2));pp=np.zeros((len(d),2))
  for f in range(4):
   extra=policy_fields(own,bet,size,f);x=np.column_stack([v['x'],extra]);np.savez_compressed(ROOT/f'{fam}_extra_fold{f}.npz',x=extra)
   if fam=='directed_transfer':y,tr,va=labels(d,a,f)
   else:yy,e,_=target(d,f);y=yy[g];tr=e[g];va=fv[g]==f
   assert not (tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(ROOT/f'{fam}_primary_fold{f}.cbm'));pp[g[va],actor[va]]=m.predict_proba(x[va],thread_count=2)[:,1];audit.append({'family':fam,'fold':f,'training_actions':int(tr.sum()),'positive_actions':int(y[tr].sum()),'validation_overlap':0,'added_features':extra.shape[1]});print('joint policy',fam,f,round(time.time()-start,1),flush=True)
  parts.append(d.select('pair_id','hand_id').with_columns(pl.Series('replacement',(pp*dw).sum(1))))
 base.join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('replacement','bg_primary').alias('bg_primary')).drop('replacement').write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps({'method':__doc__,'fits':audit},indent=2));assemble(ROOT)
if __name__=='__main__':main()
