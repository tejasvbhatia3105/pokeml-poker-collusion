import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session22_size_density');P=Path('artifacts/policy');C=pl.col
CONFIG={'sample_raises_per_table':250,'non_allin_quantile_bins':16,'allin_threshold_stack_fraction':.995,'model':'400-tree depth6 MultiClass Cat, lr .05 L2 15, original ordinary decision-state inputs, no early stopping','bins':'outer-training raise samples only; all-in is a separate atom','features':'actual-bin surprise and density, posterior CDF/tails/entropy/spread, all-in probability; scalar summaries in payoff roles and action contexts','supervision':'actual raise sizes only, no competition labels'}
def samples():
 (ROOT/'samples').mkdir(parents=True,exist_ok=True);folds=json.load(open(P/'table_folds.json'));cols=json.load(open(P/'feature_columns.json'))
 for table,f in sorted(folds.items()):
  path=ROOT/'samples'/f'{table}.parquet'
  if path.exists():continue
  a=pl.read_parquet(P/'actions'/f'{table}.parquet').filter((C('phase')=='development')&(C('action_class')==3));a=a.sample(n=min(250,len(a)),seed=2222).select('hand_id','player_id',*cols,'log_bet_ratio','amount');a.with_columns(pl.lit(table).alias('table_id'),pl.lit(f).alias('fold'),((C('amount')/C('big_blind'))/C('stack_bb').clip(1e-6)>=.995).alias('allin')).write_parquet(path,compression='zstd')
 return pl.read_parquet(list((ROOT/'samples').glob('T*.parquet'))),cols
def labels(a,edges):
 n=len(edges)-1;y=np.clip(np.searchsorted(edges,a['log_bet_ratio'].to_numpy(),side='right')-1,0,n-1);y[a['allin'].to_numpy()]=n;return y
def train():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d,cols=samples();(ROOT/'columns.json').write_text(json.dumps(cols,indent=2));X=d.select(cols).to_numpy();fv=d['fold'].to_numpy();audit=[];start=time.time()
 for f in range(4):
  tr=fv!=f;va=~tr;v=d.filter(pl.Series(tr)&~C('allin'))['log_bet_ratio'].to_numpy();edges=np.unique(np.quantile(v,np.linspace(0,1,17)));edges[0]=min(-8,float(v.min())-.1);edges[-1]=max(8,float(v.max())+.1);y=labels(d,edges);n=len(edges)-1;assert set(y[tr])==set(range(n+1));centers=np.array([d['log_bet_ratio'].to_numpy()[tr&(y==k)].mean() for k in range(n+1)]);m=CatBoostClassifier(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=15,loss_function='MultiClass',thread_count=3,random_seed=22220+f,verbose=False,allow_writing_files=False);path=ROOT/f'density_fold{f}.cbm'
  if path.exists():m.load_model(str(path))
  else:m.fit(X[tr],y[tr]);m.save_model(str(path))
  p=m.predict_proba(X[va],thread_count=3);nll=float(-np.log(p[np.arange(va.sum()),y[va]].clip(1e-12,1)).mean());prior=np.bincount(y[tr],minlength=n+1)/tr.sum();null=float(-np.log(prior[y[va]].clip(1e-12,1)).mean());(ROOT/f'metadata_fold{f}.json').write_text(json.dumps({'edges':edges.tolist(),'centers':centers.tolist(),'classes':n+1,'training_rows':int(tr.sum()),'validation_rows':int(va.sum())},indent=2));audit.append({'fold':f,'conditional_logloss':nll,'marginal_logloss':null,'allin_training_fraction':float(d['allin'].to_numpy()[tr].mean()),'bins':n});print('size density',f,round(time.time()-start,1),audit[-1],flush=True)
 (ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2))
def scalar_features(a,m,meta,cols):
 edges=np.array(meta['edges']);centers=np.array(meta['centers']);n=len(edges)-1;y=labels(a,edges);p=m.predict_proba(a.select(cols).to_numpy(),thread_count=2);ix=np.arange(len(a));mass=p[ix,y];cdf=np.cumsum(p[:,:n],axis=1);nonall=p[:,:n].sum(1);j=np.minimum(y,n-1);middle=cdf[ix,j]-.5*p[ix,j];middle=np.where(y==n,nonall+.5*p[:,-1],middle);width=np.diff(edges)[j];density=np.where(y==n,mass,mass/width);mu=p@centers;sd=np.sqrt(np.maximum(1e-8,(p*centers[None]**2).sum(1)-mu**2));actual=a['log_bet_ratio'].to_numpy();out={'surprise':-np.log(mass.clip(1e-12,1)),'log_density':np.log(density.clip(1e-12)),'cdf':middle,'two_sided_surprise':-np.log((2*np.minimum(middle,1-middle)).clip(1e-6,1)),'entropy':-(p*np.log(p.clip(1e-12,1))).sum(1),'allin_probability':p[:,-1],'standardized_residual':(actual-mu)/sd,'mean_residual':actual-mu,'posterior_sd':sd};assert np.isfinite(np.column_stack(list(out.values()))).all();return out
def build(table,q,m,meta,cols):
 a=pl.read_parquet(P/'actions'/f'{table}.parquet').join(q.select('hand_id').unique(),on='hand_id',how='semi').filter(C('action_class')==3).with_columns(C('street_no').cast(pl.Int64),((C('amount')/C('big_blind'))/C('stack_bb').clip(1e-6)>=.995).alias('allin'));fields=scalar_features(a,m,meta,cols);a=a.with_columns(*[pl.Series(k,v) for k,v in fields.items()]);mapping=pl.concat([q.select('pair_id','hand_id',C('player_1').alias('player_id'),C('player_2').alias('partner')),q.select('pair_id','hand_id',C('player_2').alias('player_id'),C('player_1').alias('partner'))]);z=a.join(mapping,on=['hand_id','player_id']);seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').select('hand_id','player_id','net_chips');st=pl.read_parquet(P/'states'/f'{table}.parquet').select('hand_id','player_id','street_no','fold_no');z=z.join(seats.rename({'net_chips':'own_net'}),on=['hand_id','player_id']).join(seats.rename({'player_id':'partner','net_chips':'partner_net'}),on=['hand_id','partner']).join(st.rename({'player_id':'partner','fold_no':'partner_fold'}),on=['hand_id','partner','street_no']);alive=C('partner_fold')>=C('action_no');facing=(C('last_aggressor')==C('partner')).fill_null(False);contexts={'alive':alive,'partner':alive&facing,'outside':alive&~facing&(C('players_active')>=3),'hu':alive&(C('players_active')==2)};expr=[]
 for role,rm in [('lower',C('own_net')<=C('partner_net')),('higher',C('own_net')>=C('partner_net'))]:
  for ctx,cm in contexts.items():
   for name in fields:
    v=C(name).filter(rm&cm);prefix=f'density_{role}_{ctx}_{name}';expr.extend([v.sum().alias(prefix+'_sum'),v.max().fill_null(0).alias(prefix+'_max'),v.min().fill_null(0).alias(prefix+'_min')])
 out=z.group_by('pair_id','hand_id').agg(expr);return q.select('pair_id','hand_id').join(out,on=['pair_id','hand_id'],how='left',validate='1:1').fill_null(0).with_columns(pl.selectors.numeric().cast(pl.Float32))
def build_features(force=False):
 (ROOT/'features').mkdir(exist_ok=True);d=hand_data();q=d.select('pair_id','hand_id','table_id').join(pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),on='pair_id',validate='m:1');cols=json.load(open(ROOT/'columns.json'));folds=json.load(open(P/'table_folds.json'));models=[];metadata=[];start=time.time()
 for f in range(4):
  m=CatBoostClassifier();m.load_model(str(ROOT/f'density_fold{f}.cbm'));models.append(m);metadata.append(json.load(open(ROOT/f'metadata_fold{f}.json')))
 for i,((table,),g) in enumerate(q.group_by('table_id')):
  path=ROOT/'features'/f'{table}.parquet'
  if path.exists() and not force:continue
  f=folds[table];out=build(table,g.drop('table_id'),models[f],metadata[f],cols);assert np.isfinite(out.select(pl.selectors.numeric()).to_numpy()).all();out.write_parquet(path,compression='zstd')
  if i%50==0:print('density features',i,round(time.time()-start,1),flush=True)
 pl.read_parquet(list((ROOT/'features').glob('T*.parquet'))).write_parquet(ROOT/'hand_features.parquet')
if __name__=='__main__':train();build_features()
