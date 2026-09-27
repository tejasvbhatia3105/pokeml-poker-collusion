import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
from session14_outcome_contrasts import ROOT,P,COLS,table_data,hand_features
from session8_data import hand_data
C=pl.col
def main():
 sample=pl.read_parquet(list((ROOT/'sample').glob('T*.parquet')));assert sample['table_id'].n_unique()==400;assert len(sample.select('hand_id','player_id','action_no').unique())==len(sample);assert not any(c in COLS for c in ['pair_id','hand_id','label','evidence','behavior_family','fold','time','target_own','target_partner','partner_fold']);folds=json.load(open(P/'table_folds.json'));assert all(folds[t]==f for t,f in sample.select('table_id','fold').unique().iter_rows());q=sample.filter(C('action_class')==0);physical=float(abs(q['target_own'].to_numpy()+q['own_committed_ratio'].to_numpy()).max());assert physical<1e-5;report={'sample_rows':len(sample),'sample_tables':400,'one_partner_per_sampled_action':True,'fold_payoff_max_error':physical,'features':len(COLS),'competition_labels_in_Q_model':False}
 d=hand_data();table=d['table_id'][0];f=folds[table];g=d.filter(C('table_id')==table).select('pair_id','hand_id').join(pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),on='pair_id',validate='m:1');m=CatBoostRegressor();m.load_model(str(ROOT/f'outcome_fold{f}.cbm'));p=CatBoostClassifier();p.load_model(str(P/f'action_fold{f}.cbm'));old=pl.read_parquet(ROOT/'features'/f'{table}.parquet').sort('pair_id','hand_id');new=hand_features(table,g,m,p).sort('pair_id','hand_id');swapped=hand_features(table,g.rename({'player_1':'player_2','player_2':'player_1'}),m,p).sort('pair_id','hand_id');replay=float(abs(old.select(pl.selectors.numeric()).to_numpy()-new.select(pl.selectors.numeric()).to_numpy()).max());swap=float(abs(new.select(pl.selectors.numeric()).to_numpy()-swapped.select(pl.selectors.numeric()).to_numpy()).max());assert replay<1e-6 and swap<1e-5;report.update({'feature_replay_error':replay,'endpoint_swap_error':swap})
 X=sample.select(COLS).to_numpy();y=sample.select('target_own','target_partner').to_numpy();va=sample['fold'].to_numpy()==0;control=CatBoostRegressor(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=20,loss_function='MultiRMSE',thread_count=3,random_seed=14140,verbose=False,allow_writing_files=False);control.fit(X[~va],y[~va]);model=CatBoostRegressor();model.load_model(str(ROOT/'outcome_fold0.cbm'));error=float(abs(control.predict(X[va],thread_count=3)-model.predict(X[va],thread_count=3)).max());assert error<1e-12;report['outcome_fold0_retrain_error']=error
 z=d.join(pl.read_parquet(ROOT/'hand_features.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');report['event_replays']={}
 for arm in ['factual','all']:
  root=ROOT/arm
  if not (root/'event_oof.parquet').exists():continue
  cols=json.load(open(root/'columns.json'));XX=z.select(cols).to_numpy();q=z.select('pair_id','hand_id','fold','behavior_family').join(pl.read_parquet(root/'event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');error=0.
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    v=((q['fold']==f)&(q['behavior_family']==fam)).to_numpy()
    for k,c in enumerate(['bg_primary','bg_secondary'],1):
     m=CatBoostClassifier();m.load_model(str(root/f'event{k}_{fam}_fold{f}.cbm'));error=max(error,float(abs(m.predict_proba(XX[v],thread_count=2)[:,1]-q[c].to_numpy()[v]).max()))
  assert error<1e-12;report['event_replays'][arm]={'heads':24,'max_error':error}
 (ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
