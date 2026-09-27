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
from session26_exact_fold_witness import actions
from session37_bet_fold import paired_features
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session41_isolation_bet_fold');BASE=Path('artifacts/evidence_session38_soft_bet_fold/paired_soft')
def data():
 full=hand_data();d=full.filter(C('behavior_family')=='coordinated_isolation').drop('row').with_row_index('row');a,ac=actions(d);a=a.with_row_index('action_row');assert a.select('pair_id','hand_id').n_unique()==len(a);return full,d,a,ac
def designs(d,a,ac,extra):
 cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));hx=d.select(cfg['event']).to_numpy();g=a['row'].to_numpy();raw=np.full((len(d),len(ac)+1),-2.);raw[:,-1]=0;raw[g,:-1]=a.select(ac).to_numpy();raw[g,-1]=1;ec=[c for c in extra.columns if c!='action_row'];ex=np.full((len(d),len(ec)),-2.);ex[g]=extra.select(ec).to_numpy();return {'raw_fold':np.column_stack([hx,raw]),'paired_fold':np.column_stack([hx,raw,ex])},cfg['event'],ec
def main():
 ROOT.mkdir(exist_ok=True);full,d,a,ac=data();a.write_parquet(ROOT/'fold_actions.parquet');extra,align=paired_features(d,a);extra.write_parquet(ROOT/'action_features.parquet');align.write_parquet(ROOT/'alignment.parquet');xs,hc,ec=designs(d,a,ac,extra);pred={k:np.zeros((len(d),2)) for k in xs};audit=[];start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'hand_columns':hc,'fold_columns':ac,'paired_columns':ec,'missing':'-2 plus explicit fold-present indicator','models':'same Cat400 depth5 original two event-head seeds','targets':'unchanged original censored isolation labels; absence of a partner fold does not force event probability0','baseline':str(BASE)},indent=2));dv=d['fold'].to_numpy();familymask=full['behavior_family'].to_numpy()=='coordinated_isolation'
 for f in range(4):
  p1,p2,e1,e2,_=targets(full,f,'coordinated_isolation');y=np.column_stack([p1,p2])[familymask];es=np.column_stack([e1,e2])[familymask];va=dv==f
  for kind,x in xs.items():
   root=ROOT/kind;root.mkdir(exist_ok=True)
   for k in range(2):
    tr=es[:,k];assert not (tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f+k,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr,k]);m.save_model(str(root/f'event{k+1}_fold{f}.cbm'));pred[kind][va,k]=m.predict_proba(x[va],thread_count=2)[:,1];audit.append({'fold':f,'kind':kind,'head':k+1,'training_hands':int(tr.sum()),'positive_hands':int(y[tr,k].sum()),'validation_overlap':0})
   print('isolation paired',f,kind,round(time.time()-start,1),flush=True)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));base=pl.read_parquet(BASE/'event_oof.parquet')
 for kind,pp in pred.items():
  root=ROOT/kind;q=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',pp[:,0]),pl.Series('new_secondary',pp[:,1]));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary'),pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_primary','new_secondary').write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
