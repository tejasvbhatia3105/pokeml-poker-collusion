import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session22_size_density import ROOT,P,labels,build
from session8_data import hand_data
C=pl.col
def main():
 sample=pl.read_parquet(list((ROOT/'samples').glob('T*.parquet')));cols=json.load(open(ROOT/'columns.json'));fv=sample['fold'].to_numpy();assert sample['table_id'].n_unique()==400;assert len(sample)==100000;models=[];metadata=[];reports=[];audit=json.load(open(ROOT/'training_audit.json'))
 for f in range(4):
  meta=json.load(open(ROOT/f'metadata_fold{f}.json'));metadata.append(meta);tr=fv!=f;va=~tr;v=sample.filter(pl.Series(tr)&~C('allin'))['log_bet_ratio'].to_numpy();edges=np.unique(np.quantile(v,np.linspace(0,1,17)));edges[0]=min(-8,float(v.min())-.1);edges[-1]=max(8,float(v.max())+.1);assert np.array_equal(edges,meta['edges']);y=labels(sample,edges);m=CatBoostClassifier();m.load_model(str(ROOT/f'density_fold{f}.cbm'));models.append(m);p=m.predict_proba(sample.select(cols).to_numpy()[va],thread_count=2);assert np.allclose(p.sum(1),1,atol=1e-12);nll=float(-np.log(p[np.arange(va.sum()),y[va]].clip(1e-12,1)).mean());err=abs(nll-audit[f]['conditional_logloss']);assert err<1e-12;cum=p[:,:-1].cumsum(1)-.5*p[:,:-1];allin_mid=1-.5*p[:,-1];assert np.all(cum[:,-1]<=allin_mid+1e-12);assert allin_mid.min()>=.5-1e-12;reports.append({'fold':f,'bin_edges_from_training_only':True,'probability_sum_max_error':float(abs(p.sum(1)-1).max()),'saved_validation_logloss_error':err})
 d=hand_data();counts=d.group_by('table_id').agg(C('pair_id').n_unique().alias('n')).sort('n',descending=True);table=counts['table_id'][0];folds=json.load(open(P/'table_folds.json'));f=folds[table];q=d.filter(C('table_id')==table).select('pair_id','hand_id').join(pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),on='pair_id');orig=pl.read_parquet(ROOT/'features'/f'{table}.parquet').sort('pair_id','hand_id');v=build(table,q,models[f],metadata[f],cols).sort('pair_id','hand_id');swap=build(table,q.rename({'player_1':'player_2','player_2':'player_1'}),models[f],metadata[f],cols).sort('pair_id','hand_id');replay=float(abs(orig.select(pl.selectors.numeric()).to_numpy()-v.select(pl.selectors.numeric()).to_numpy()).max());swaperr=float(abs(v.select(pl.selectors.numeric()).to_numpy()-swap.select(pl.selectors.numeric()).to_numpy()).max());assert replay<1e-6 and swaperr<1e-5;out={'sample_rows':len(sample),'folds':reports,'corrected_cdf_feature_replay_error':replay,'endpoint_swap_error':swaperr,'event_replays':{}};extra=pl.read_parquet(ROOT/'hand_features.parquet');z=d.join(extra,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
 for arm in ['density','location_spread']:
  root=ROOT/arm
  if not (root/'event_oof.parquet').exists():continue
  cs=json.load(open(root/'columns.json'));X=z.select(cs).to_numpy();q=z.select('pair_id','hand_id','fold','behavior_family').join(pl.read_parquet(root/'event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    va=((q['fold']==f)&(q['behavior_family']==fam)).to_numpy()
    for k,c in enumerate(['bg_primary','bg_secondary'],1):
     m=CatBoostClassifier();m.load_model(str(root/f'event{k}_{fam}_fold{f}.cbm'));err=max(err,float(abs(m.predict_proba(X[va],thread_count=2)[:,1]-q[c].to_numpy()[va]).max()))
  assert err<1e-12;out['event_replays'][arm]={'models':24,'max_error':err}
 (ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
