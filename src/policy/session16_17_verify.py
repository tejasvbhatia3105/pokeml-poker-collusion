import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl,joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data
from session16_integrated import data
from session17_grounded_tiers import check,assignments
from session19_grounded_directed import options
C=pl.col
def main():
 report={'grounded_math':check()};d,cols=data();root=Path('artifacts/evidence_session16_integrated');q=d.select('pair_id','hand_id','fold','behavior_family').join(pl.read_parquet(root/'heads_oof.parquet').drop('fold','time','behavior_family'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');X=d.select(cols).to_numpy();err=0.
 with threadpool_limits(limits=2):
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    use=((q['fold']==f)&(q['behavior_family']==fam)).to_numpy()
    for name,path in [('new_base_cat',root/f'base_{fam}_fold{f}.cbm'),('bg_primary',root/f'event1_{fam}_fold{f}.cbm'),('bg_secondary',root/f'event2_{fam}_fold{f}.cbm')]:
     m=CatBoostClassifier();m.load_model(str(path));err=max(err,float(abs(m.predict_proba(X[use],thread_count=2)[:,1]-q[name].to_numpy()[use]).max()))
    m=joblib.load(root/f'base_{fam}_fold{f}.joblib');err=max(err,float(abs(m.predict_proba(X[use])[:,1]-q['new_base_hist'].to_numpy()[use]).max()))
  assert err<1e-12;report['integrated']={'models':48,'prediction_max_error':err,'feature_count':len(cols),'note':'base refit uses hand_data chronological order; legacy Cat retrieval fit sorted pair/hand ID, an additional recipe difference; this rejected arm is not a pure feature ablation'}
  for dirname,fam,levels,fn in [('evidence_session17_grounded','soft_play',3,assignments),('evidence_session19_directed','directed_transfer',2,options)]:
   root=Path('artifacts')/dirname;d=hand_data().filter(C('behavior_family')==fam).drop('row').with_row_index('local_row');cs=json.load(open(root/'columns.json'));X=d.select(cs['event']).to_numpy();XT=d.select(cs['type']).to_numpy();fv=d['fold'].to_numpy();q=d.select('pair_id','hand_id').join(pl.read_parquet(root/'heads_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.;fold_audit=[]
   for f in range(4):
    typ=CatBoostClassifier();typ.load_model(str(root/f'type_fold{f}.cbm'));tp=typ.predict_proba(XT,thread_count=2);saved={r['pair_id']:r for r in json.load(open(root/f'assignments_fold{f}.json'))};known=0;skipped=[]
    for (pid,),g in d.group_by('pair_id'):
     if g['fold'][0]==f:assert pid not in saved;continue
     ix,valid=fn(g)
     if not valid:assert pid not in saved;skipped.append(pid);continue
     if len(valid)==1:known+=len(ix)
     expected=valid[int(np.argmax([np.log(tp[ix,c].clip(1e-8,1)).sum() for c in valid]))];assert expected.tolist()==saved[pid]['chosen']
    va=fv==f
    for k in range(levels):
     m=CatBoostClassifier();m.load_model(str(root/f'event{k}_fold{f}.cbm'));err=max(err,float(abs(m.predict_proba(X[va],thread_count=2)[:,1]-q[f'cat_{k}'].to_numpy()[va]).max()));m=joblib.load(root/f'event{k}_fold{f}.joblib');err=max(err,float(abs(m.predict_proba(X[va])[:,1]-q[f'hist_{k}'].to_numpy()[va]).max()))
    fold_audit.append({'fold':f,'unique_training_type_rows':known,'incompatible_training_pairs':skipped,'heldout_pairs_in_assignment_cache':0})
   assert err<1e-12;report[dirname]={'event_models':8*levels,'prediction_max_error':err,'saved_assignment_replay':True,'folds':fold_audit}
 Path('artifacts/evidence_session16_19_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
