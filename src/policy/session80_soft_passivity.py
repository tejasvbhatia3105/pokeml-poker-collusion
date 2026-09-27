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
from session8_data import hand_data,targets
from session58_pressure_comparison import compute,COLS as CURRENT
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session80_soft_passivity');C=pl.col
def data():
 full=hand_data();d=full.filter(C('behavior_family')=='soft_play').drop('row').with_row_index('row');raw=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet').filter(C('behavior_family')=='soft_play');ac=[c for c in raw.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family','time']];a=raw.filter(((C('action_class')==2)&C('facing_partner')&C('mw_alive'))|((C('action_class')==1)&(C('players_active')==2)&C('mw_alive'))).select('pair_id','hand_id',*ac).join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='m:1').sort('row','street_no','action_no');return full,d,a,ac
def features(d,a,ac):
 ex,rawaudit=compute(d,a);q=a.with_columns(*[pl.Series(c,ex[:,i]) for i,c in enumerate(CURRENT)]);cols=ac+CURRENT;out=[];names=[]
 for kind,k in [('call',2),('check',1)]:
  local=q.filter(C('action_class')==k);cc=[kind+'_'+c+'_'+s for s in ['mean','min','max'] for c in cols];z=local.group_by('pair_id','hand_id').agg(*[getattr(C(c),s)().alias(kind+'_'+c+'_'+s) for s in ['mean','min','max'] for c in cols],pl.len().alias(kind+'_count'));z=d.select('pair_id','hand_id').join(z,on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(C(kind+'_count').fill_null(0));out.append(z.select(cc+[kind+'_count']).fill_null(-2).to_numpy());names.extend(cc+[kind+'_count'])
 return np.column_stack(out).astype(np.float32),names,rawaudit
def main():
 ROOT.mkdir(exist_ok=True);full,d,a,ac=data();extra,names,rawaudit=features(d,a,ac);a.write_parquet(ROOT/'actions.parquet');np.savez_compressed(ROOT/'features.npz',x=extra);cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));hx=d.select(cfg['event']).to_numpy();x=np.column_stack([hx,extra]);fv=d['fold'].to_numpy();mask=full['behavior_family'].to_numpy()=='soft_play';pred=np.zeros(len(d));control=np.zeros(len(d));audit=[];start=time.time();(ROOT/'columns.json').write_text(json.dumps({'hand':cfg['event'],'action':ac,'added':names},indent=2));(ROOT/'feature_audit.json').write_text(json.dumps(rawaudit,indent=2))
 for f in range(4):
  p1,p2,e1,e2,va=targets(full,f,'soft_play');y=p2[mask];tr=e2[mask];va=fv==f;assert not (tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6312+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(ROOT/f'secondary_fold{f}.cbm'));pred[va]=m.predict_proba(x[va],thread_count=2)[:,1];m=CatBoostClassifier();m.load_model(f'artifacts/evidence_session6/priority_ordered_event2_soft_play_fold{f}.cbm');control[va]=m.predict_proba(hx[va],thread_count=2)[:,1];audit.append({'fold':f,'training_hands':int(tr.sum()),'positive_hands':int(y[tr].sum()),'validation_overlap':0});print('soft passivity',f,round(time.time()-start,1),flush=True)
 base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');old=d.select('pair_id','hand_id').join(base,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');error=float(abs(old['bg_secondary'].to_numpy()-control).max());assert error<1e-12;q=d.select('pair_id','hand_id').with_columns(pl.Series('replacement',pred));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('replacement','bg_secondary').alias('bg_secondary')).drop('replacement').write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps({'method':__doc__,'control_replay_error':error,'fits':audit},indent=2));assemble(ROOT)
if __name__=='__main__':main()
