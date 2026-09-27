\
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
from catboost import CatBoostClassifier,Pool
from scipy.special import softmax
import session94_player_kernel as s
ROOT=Path('artifacts/evidence_session95_learned_player_kernel');C=pl.col
CONFIG={'iterations':250,'depth':5,'learning_rate':.04,'l2_leaf_reg':30,'seed':9500,'arms':['current','context'],'context_method':s.CONFIG,'reference':'strict nested84; outer f plus native h excluded for training, mean3 excluding f at validation','base_fields':s.PC+['reference_logp0','reference_logp1','reference_logp2','reference_logp3'],'extra_fields':[f'{a}_log_offset{k}' for a in ['intercept','kernel'] for k in range(4)]+['log_context_count','log_kernel_mass'],'evidence_labels':False}
def legal(p,x):
 p=p.copy();call=x[:,s.PC.index('call_bb')]>0;p[call,1]=0;p[~call,0]=0;p[~call,2]=0;p/=p.sum(1)[:,None];return p
def fields(q,h,p,ph):
 z,mass=s.calibrate(q,h,p,ph);extra=np.column_stack([np.log(z[k].clip(1e-7))-np.log(p.clip(1e-7)) for k in ['intercept','kernel']]+[np.log1p(mass)]);return extra.astype(np.float32)
def prepare():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));models=s.load_models();tf=json.load(open('artifacts/policy/table_folds.json'));xx=[];yy=[];fv=[];meta=[];extra=[[] for _ in range(4)];prior=[[] for _ in range(4)];start=time.time()
 for i,path in enumerate(sorted(Path('artifacts/policy/actions').glob('*.parquet'))):
  a=pl.read_parquet(path).filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');q,h=s.select(a);native=tf[path.stem];x=q.select(s.PC).to_numpy();hx=h.select(s.PC).to_numpy();p={key:legal(m.predict_proba(x,thread_count=2),x) for key,m in models.items() if native in key};ph={key:legal(m.predict_proba(hx,thread_count=2),hx) for key,m in models.items() if native in key}
                                                                             
  rawp={key:models[key].predict_proba(x,thread_count=2) for key in p};rawph={key:models[key].predict_proba(hx,thread_count=2) for key in p}
  for f in range(4):
   pp=legal(sum(rawp.values())/3,x) if f==native else p[tuple(sorted([f,native]))];hp=legal(sum(rawph.values())/3,hx) if f==native else ph[tuple(sorted([f,native]))];prior[f].append(np.log(pp.clip(1e-7)).astype(np.float32));extra[f].append(fields(q,h,pp,hp))
   if f==native:
    old=pl.read_parquet(s.ROOT/'tables'/path.name);np.testing.assert_array_equal(pp,old.select([f'reference_{j}' for j in range(4)]).to_numpy())
  xx.append(x);yy.append(q['action_class'].to_numpy());fv.append(np.full(len(q),native,np.int8));meta.append(q.select('table_id','hand_id','player_id','action_no','source_row'))
  if (i+1)%50==0:print('learned player inputs',i+1,round(time.time()-start,1),flush=True)
 np.savez_compressed(ROOT/'sample.npz',x=np.concatenate(xx),y=np.concatenate(yy),fold=np.concatenate(fv));pl.concat(meta).write_parquet(ROOT/'sample_keys.parquet')
 for f in range(4):np.savez_compressed(ROOT/f'inputs_fold{f}.npz',prior=np.concatenate(prior[f]),extra=np.concatenate(extra[f]))
def predict(model,x,prior):
 z=prior+model.predict(x,prediction_type='RawFormulaVal',thread_count=2);call=x[:,s.PC.index('call_bb')]>0;z[call,1]=-30;z[~call,0]=-30;z[~call,2]=-30;return legal(softmax(z,axis=1),x)
def train():
 z=np.load(ROOT/'sample.npz');raw=z['x'];y=z['y'];fv=z['fold'];start=time.time();audit=[]
 for f in range(4):
  v=np.load(ROOT/f'inputs_fold{f}.npz');prior=v['prior'];extra=v['extra'];base=np.column_stack([raw,prior]);tr=fv!=f;va=~tr;baseline=legal(softmax(prior[va],axis=1),raw[va]);np.save(ROOT/f'reference_fold{f}.npy',baseline)
  for kind in CONFIG['arms']:
   x=np.column_stack([base,extra]) if kind=='context' else base;m=CatBoostClassifier(iterations=CONFIG['iterations'],depth=CONFIG['depth'],learning_rate=CONFIG['learning_rate'],l2_leaf_reg=CONFIG['l2_leaf_reg'],random_seed=CONFIG['seed']+f,loss_function='MultiClass',thread_count=2,verbose=False,allow_writing_files=False);m.fit(Pool(x[tr],y[tr],baseline=prior[tr]));m.save_model(str(ROOT/f'{kind}_fold{f}.cbm'));p=predict(m,x[va],prior[va]);np.save(ROOT/f'{kind}_fold{f}.npy',p);audit.append({'fold':f,'kind':kind,'training_actions':int(tr.sum()),'validation_actions':int(va.sum()),'validation_overlap':0,'heldout_logloss':float(-np.log(p[np.arange(len(p)),y[va]].clip(1e-7)).mean())});print('learned player',f,kind,audit[-1]['heldout_logloss'],round(time.time()-start,1),flush=True)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));compare()
def compare():
 z=np.load(ROOT/'sample.npz');meta=pl.read_parquet(ROOT/'sample_keys.parquet');fv=z['fold'];y=z['y'];parts=[];cols=['reference',*CONFIG['arms']]
 for f in range(4):
  va=fv==f;loss={}
  for k in cols:
   p=np.load(ROOT/f'{k}_fold{f}.npy');loss[k]=-np.log(p[np.arange(len(p)),y[va]].clip(1e-7))
  parts.append(meta.filter(pl.Series(va)).with_columns(pl.lit(f).alias('fold'),*[pl.Series(k,v) for k,v in loss.items()]))
 q=pl.concat(parts);pool=q.group_by('table_id').agg(C(cols).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(9512).integers(0,len(pool),(5000,len(pool)));report={'metric':'heldout gameplay action logloss, not evidence MAP or Kaggle score','actions':len(q),'pools':len(pool),'mean_logloss':{k:q[k].mean() for k in cols},'folds':q.group_by('fold').agg(C(cols).mean()).sort('fold').to_dicts(),'deltas':{}}
 for a,b in [('current','reference'),('context','reference'),('context','current')]:
  diff=pool[a].to_numpy()-pool[b].to_numpy();boot=diff[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report['deltas'][f'{a}_minus_{b}']={'mean':float(q[a].mean()-q[b].mean()),'pool_CI95':np.quantile(boot,[.025,.975]).tolist()}
 (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));pool.write_parquet(ROOT/'pool_losses.parquet');print(json.dumps(report,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 else:train()
