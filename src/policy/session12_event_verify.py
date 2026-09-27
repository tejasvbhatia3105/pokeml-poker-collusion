import os,json,sys
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
C=pl.col

def main():
 root=Path(sys.argv[1]);d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];assert not set(cols)&{'evidence','evidence_rank','subtype','fold','pair_id','hand_id'};X=d.select(cols).to_numpy();saved=pl.read_parquet(root/'event_oof.parquet');err=0.;models=0
 for f in range(4):
  for fam in ['directed_transfer','soft_play','coordinated_isolation']:
   va=(d['fold'].to_numpy()==f)&(d['behavior_family'].to_numpy()==fam);g=d.filter(pl.Series(va)).select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
   for k in [1,2]:
    m=CatBoostClassifier();m.load_model(str(root/f'event{k}_{fam}_fold{f}.cbm'));p=m.predict_proba(X[va],thread_count=3)[:,1];want=g['bg_primary' if k==1 else 'bg_secondary'].to_numpy();err=max(err,float(abs(p-want).max()));models+=1
 control=Path('artifacts/evidence_session12/background/positive_control_check.json')
 if not control.exists():
  a,b,ea,eb,va=targets(d,0,'directed_transfer');m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=3,random_seed=6311,verbose=False,allow_writing_files=False);m.fit(X[ea],a[ea]);old=CatBoostClassifier();old.load_model('artifacts/evidence_session6/priority_ordered_event1_directed_transfer_fold0.cbm');delta=float(abs(m.predict_proba(X[va],thread_count=3)-old.predict_proba(X[va],thread_count=3)).max());control.write_text(json.dumps({'positive_control_probability_replay_error':delta,'head':'directed primary fold0; exact archived inputs/targets/config'},indent=2));assert delta<1e-12
 extra={}
 if (root/'negative_features.parquet').exists():
  n=pl.read_parquet(root/'negative_features.parquet',columns=['pair_id','hand_id','fold']);known=pl.read_csv('data/development_labels.csv').filter(C('label')==0);assert set(n['pair_id'])==set(known['pair_id']);extra['confirmed_negative_pairs']=n['pair_id'].n_unique();extra['negative_hands']=len(n)
 if (root/'future_features.parquet').exists():
  n=pl.read_parquet(root/'future_features.parquet',columns=['pair_id','hand_id','fold','phase']);known=pl.read_csv('data/development_labels.csv').filter(C('label')==1);assert set(n['pair_id'])<=set(known['pair_id']);assert set(n['phase'])=={'evaluation'};assert not len(n.join(d.select('pair_id','hand_id'),on=['pair_id','hand_id']));extra['later_training_relationships']=n['pair_id'].n_unique();extra['future_hands']=len(n)
 assert err<1e-12;report={'models_replayed':models,'max_event_replay_error':err,'unchanged_positive_control':json.loads(control.read_text()),'excluded_features':'labels/rank/subtype/IDs/fold excluded','auxiliary_features':extra};(root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
