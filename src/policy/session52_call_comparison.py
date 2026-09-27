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
from session39_bet_call import data,targets,OLD
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session52_call_comparison');C=pl.col
COLS=['call_count','postflop_count','order_mean','order_min','order_max','rank_gap_mean','rank_gap_min','rank_gap_max','weaker_log_amount_sum','stronger_log_amount_sum']
def features(d,a):
 ex=pl.read_parquet('artifacts/evidence_session39_bet_call/action_features.parquet').sort('action_row');assert np.array_equal(ex['action_row'].to_numpy(),a['action_row'].to_numpy());gap=a['made_category'].to_numpy().astype(np.float64)+a['made_kicker'].to_numpy()-ex['bet_made_category'].to_numpy()-ex['bet_made_kicker'].to_numpy();post=a['street_no'].to_numpy()>0;sg=np.sign(gap);out=np.zeros((len(d),2,len(COLS)),np.float32);out[:,:,2:8]=-2
 for (row,actor),g in a.group_by(['row','actor']):
  ix=g['action_row'].to_numpy();j=ix[post[ix]];v=out[row,actor];v[0]=len(ix);v[1]=len(j)
  if len(j):v[2:5]=[sg[j].mean(),sg[j].min(),sg[j].max()];v[5:8]=[gap[j].mean(),gap[j].min(),gap[j].max()];v[8]=a['log_amount_bb'].to_numpy()[j][sg[j]<0].sum();v[9]=a['log_amount_bb'].to_numpy()[j][sg[j]>0].sum()
 return out
def audit_calls(d,a):
 labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');records=[];total=allin=0
 for (table,),g in d.group_by('table_id'):
  q=g.select('pair_id','hand_id').join(labs,on='pair_id');mapping=pl.concat([q.select('pair_id','hand_id',C('player_1').alias('player_id'),C('player_2').alias('partner'),pl.lit(0).alias('actor')),q.select('pair_id','hand_id',C('player_2').alias('player_id'),C('player_1').alias('partner'),pl.lit(1).alias('actor'))]);raw=pl.read_parquet(f'artifacts/compact/actions/table_id={table}/*.parquet').join(g.select('hand_id').unique(),on='hand_id',how='semi');pc=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').with_columns(C('action_no').cast(pl.Int64)).join(g.select('hand_id').unique(),on='hand_id',how='semi');cc=pl.when(C('amount')>C('to_call')).then(3).when(C('action')=='fold').then(0).when(C('action')=='check').then(1).otherwise(2);raw=raw.with_columns(cc.alias('correct_class'));z=pc.select('hand_id','action_no','action_class').join(raw.select('hand_id',C('action_no').cast(pl.Int64),'correct_class','action'),on=['hand_id','action_no'],validate='1:1');assert (z['action_class']==z['correct_class']).all();allin+=len(z.filter((C('action')=='all_in')&(C('correct_class')==2)));expected=pc.join(mapping,on=['hand_id','player_id']).filter((C('action_class')==2)&(C('last_aggressor')==C('partner')));keys=['pair_id','hand_id','actor','action_no'];cached=a.join(g.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');assert set(expected.select(keys).iter_rows())==set(cached.select(keys).iter_rows());total+=len(z);records.append({'table':table,'cached_calls':len(cached),'expected_calls':len(expected)})
 return {'raw_action_classes_checked':total,'allin_calls_checked':allin,'call_cache_exact_complete_against_raw_policy_actions':True,'tables':records}
def main():
 ROOT.mkdir(exist_ok=True);d,a,cfg=data();AX=np.load(OLD/'actor_features.npz')['hand'];XX=features(d,a);np.savez_compressed(ROOT/'features.npz',x=XX);(ROOT/'feature_audit.json').write_text(json.dumps(audit_calls(d,a),indent=2));(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'columns':COLS,'schedule':'Cat400 depth5 lr.035 L2 8 seed6312+11fold','baseline':'R32 current-only','eligibility':'all original25 secondary hand targets, no call support gate'},indent=2));hx=d.select(cfg['hand']).to_numpy();fv=d['fold'].to_numpy();pred={k:np.zeros((len(d),2)) for k in ['control','comparison']};audit=[];start=time.time()
 for f in range(4):
  y,tr,donor=targets(d,f);vi=np.flatnonzero(fv==f);ti=np.flatnonzero(tr);assert not tr[vi].any()
  for kind in pred:
   root=ROOT/kind;root.mkdir(exist_ok=True);x=np.column_stack([hx[ti],AX[ti,donor[ti]]]);x=np.column_stack([x,XX[ti,donor[ti]]]) if kind=='comparison' else x;m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6312+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x,y[ti]);m.save_model(str(root/f'secondary_fold{f}.cbm'))
   for actor in range(2):
    v=np.column_stack([hx[vi],AX[vi,actor]]);v=np.column_stack([v,XX[vi,actor]]) if kind=='comparison' else v;pred[kind][vi,actor]=m.predict_proba(v,thread_count=2)[:,1]
   audit.append({'fold':f,'kind':kind,'training_hands':len(ti),'positive_hands':int(y[ti].sum()),'validation_overlap':0});print('call comparison',f,kind,round(time.time()-start,1),flush=True)
 dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();base=pl.read_parquet('artifacts/evidence_session50_matchup/current/event_oof.parquet')
 for kind,pp in pred.items():
  root=ROOT/kind;d.select('pair_id','hand_id').with_columns(pl.Series('actor0',pp[:,0]),pl.Series('actor1',pp[:,1])).write_parquet(root/'conditional_oof.parquet');q=d.select('pair_id','hand_id').with_columns(pl.Series('new_secondary',(pp*dw).sum(1)));out=base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_secondary');out.write_parquet(root/'event_oof.parquet');assemble(root)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
