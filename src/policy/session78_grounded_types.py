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
from sklearn.metrics import log_loss,accuracy_score,balanced_accuracy_score
from session8_data import hand_data
from session62_grounded_list_boost import grounded
from session55_current_targets import soft_support,direct,soft
ROOT=Path('artifacts/evidence_session78_grounded_types');C=pl.col
def main():
 ROOT.mkdir(exist_ok=True);full=hand_data();gx,gcols=grounded(full);cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));parts=[];audit=[];reports={};start=time.time()
 for fam in ['directed_transfer','soft_play']:
  mask=full['behavior_family'].to_numpy()==fam;d=full.filter(C('behavior_family')==fam).drop('row').with_row_index('row');x=gx[mask];xt=d.select(cfg['type']).to_numpy();fv=d['fold'].to_numpy();sub=np.where(d['subtype'].to_numpy()>0,2-d['subtype'].to_numpy(),-1);nc=2 if fam=='directed_transfer' else 3
  if fam=='soft_play':sub,known,_=soft_support(d,np.ones(len(d),bool))
  else:known=sub>=0
  preds={k:np.zeros((len(d),nc)) for k in ['original','grounded','augmented']};(ROOT/f'{fam}_columns.json').write_text(json.dumps({'original':cfg['type'],'grounded':gcols},indent=2));changes=[]
  for f in range(4):
   tr=fv!=f;va=~tr;train=tr&known
   for kind in preds:
    X=xt if kind=='original' else x if kind=='grounded' else np.column_stack([xt,x]);m=CatBoostClassifier()
    if kind=='original':m.load_model(f'artifacts/evidence_session6/priority_ordered_type_{fam}_fold{f}.cbm' if nc==2 else f'artifacts/evidence_session17_grounded/type_fold{f}.cbm')
    else:
     m=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,random_seed=6210+f,thread_count=2,verbose=False,allow_writing_files=False,loss_function='Logloss' if nc==2 else 'MultiClass');m.fit(X[train],sub[train]);m.save_model(str(ROOT/f'{fam}_{kind}_fold{f}.cbm'))
    pp=m.predict_proba(X,thread_count=2);preds[kind][va]=pp[va]
    if nc==2:y,e,_,r=direct(d,tr,pp[:,1])
    else:y,e,r=soft(d,tr,pp)
    if kind=='original':oy=y.copy();oe=e.copy()
    else:changes.append({'fold':f,'kind':kind,'changed_training_targets':int((y!=oy).sum()),'changed_eligibility':int((e!=oe).sum())})
    audit.append({'family':fam,'fold':f,'kind':kind,'training_known':int(train.sum()),'validation_known':int((known&va).sum()),'training_validation_overlap':int((train&va).sum())})
   print(fam,f,round(time.time()-start,1),flush=True)
  metrics={}
  for kind,p in preds.items():metrics[kind]={'accuracy':accuracy_score(sub[known],p[known].argmax(1)),'balanced_accuracy':balanced_accuracy_score(sub[known],p[known].argmax(1)),'logloss':log_loss(sub[known],p[known],labels=list(range(nc))),'fold_logloss':[log_loss(sub[known&(fv==f)],p[known&(fv==f)],labels=list(range(nc))) if (known&(fv==f)).any() else None for f in range(4)]}
  reports[fam]={'known_hands':int(known.sum()),'known_pairs':d.filter(pl.Series(known))['pair_id'].n_unique(),'class_counts':np.bincount(sub[known],minlength=nc).tolist(),'metrics':metrics,'target_changes':changes};d.select('pair_id','hand_id','fold').with_columns(pl.Series('known_type',sub),*[pl.Series(kind+'_'+str(k),p[:,k]) for kind,p in preds.items() for k in range(nc)]).write_parquet(ROOT/f'{fam}_oof.parquet')
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'report.json').write_text(json.dumps({'method':__doc__,'families':reports},indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
