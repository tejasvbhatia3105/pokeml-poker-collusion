\
\
\
\
\
\
\
\
import os,json,time,hashlib,itertools
os.environ.setdefault('POLARS_MAX_THREADS','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session84_nested_joint_policy import paths
ROOT=Path('artifacts/evidence_session94_player_kernel');C=pl.col
CONFIG={'sample_seed':9401,'query_street_caps':[200,100,50,50],'context_cap_per_player':512,'query_hands':'time_index modulo5==0','context':'other development hands; retrospective same-phase context','pseudo_count':20.,'kernel_fields':['equity','pot_odds','log1p(pot_bb)','log1p(stack_bb)','players_active'],'kernel_scales':[.2,.15,1.5,1.5,2.],'exact_kernel_keys':['player_id','street_no','call_bb>0'],'arms':['reference','intercept','kernel'],'styles':'recomputed from full nonquery hands, leave own hand/street out for context rows','reference':'mean3 session84 two-fold-excluded policies; query-pool fold excluded by all3','evidence_labels_used':False}
PC=json.load(open('artifacts/policy/feature_columns.json'))
def restyle(a,r):
 \
 keys=['player_id','street_no'];local=keys+['time_bin'];hk=local+['hand_id'];cnt=a.group_by(hk).agg(pl.len().alias('_hn'),*[(C('action_class')==k).sum().alias(f'_h{k}') for k in range(4)]);glob=cnt.group_by(keys).agg(C('_hn').sum().alias('_gn'),*[C(f'_h{k}').sum().alias(f'_g{k}') for k in range(4)]);loc=cnt.group_by(local).agg(C('_hn').sum().alias('_ln'),*[C(f'_h{k}').sum().alias(f'_l{k}') for k in range(4)]);q=r.join(glob,on=keys,how='left',validate='m:1',maintain_order='left').join(loc,on=local,how='left',validate='m:1',maintain_order='left').join(cnt,on=hk,how='left',validate='m:1',maintain_order='left');aux=[c for c in q.columns if c.startswith('_')];q=q.with_columns(C(aux).fill_null(0))
 q=q.with_columns(*[((C(f'_g{k}')-C(f'_h{k}')+1)/(C('_gn')-C('_hn')+4)).cast(pl.Float32).alias(f'style_{k}') for k in range(4)]);q=q.with_columns(*[((C(f'_l{k}')-C(f'_h{k}')+20*C(f'style_{k}'))/(C('_ln')-C('_hn')+20)).cast(pl.Float32).alias(f'local_style_{k}') for k in range(4)]);return q.drop(aux)
def select(a):
 candidates=a.filter(C('time_index')%5==0);q=pl.concat([z.sample(n=min(cap,len(z)),seed=CONFIG['sample_seed']) for st,cap in enumerate(CONFIG['query_street_caps']) if len(z:=candidates.filter(C('street_no')==st))]).sort('source_row');history=a.filter(C('time_index')%5!=0);h=pl.concat([z.sample(n=min(len(z),CONFIG['context_cap_per_player']),seed=CONFIG['sample_seed']) for _,z in history.group_by('player_id',maintain_order=True)]).sort('source_row');assert not len(q.select('hand_id').join(h.select('hand_id'),on='hand_id',how='semi'));return restyle(history,q),restyle(history,h)
def reference(x,fold,models):
 p=sum(m.predict_proba(x,thread_count=2) for key,m in models.items() if fold in key)/3;assert sum(fold in key for key in models)==3;call=x[:,PC.index('call_bb')]>0;p[call,1]=0;p[~call,0]=0;p[~call,2]=0;p/=p.sum(1)[:,None];return p
def kernel_design(a):
 x=np.column_stack([a['equity'],a['pot_odds'],np.log1p(a['pot_bb']),np.log1p(a['stack_bb']),a['players_active']]);return x/np.array(CONFIG['kernel_scales'])
def calibrate(q,h,p,ph):
 x=kernel_design(q);hx=kernel_design(h);y=h['action_class'].to_numpy().astype(int);one=np.eye(4)[y];prob={k:p.copy() for k in ['intercept','kernel']};mass=np.zeros((len(q),2));qkey=q.with_row_index('i').with_columns((C('call_bb')>0).alias('facing'));hkey=h.with_row_index('i').with_columns((C('call_bb')>0).alias('facing'));lookup={key:g['i'].to_numpy() for key,g in hkey.group_by('player_id','street_no','facing')}
 for key,g in qkey.group_by('player_id','street_no','facing'):
  qi=g['i'].to_numpy();hi=lookup.get(key)
  if hi is None:continue
  diff=x[qi,None]-hx[None,hi];w=np.exp(-.5*(diff*diff).sum(2))
  for j,(kind,ww) in enumerate([('intercept',np.ones_like(w)),('kernel',w)]):
   observed=ww@one[hi];expected=ww@ph[hi];smooth=CONFIG['pseudo_count']*p[qi];ratio=(observed+smooth)/(expected+smooth+1e-30);z=p[qi]*ratio;z/=z.sum(1)[:,None];prob[kind][qi]=z;mass[qi,j]=ww.sum(1)
 return prob,mass
def load_models():
 models={}
 for key in itertools.combinations(range(4),2):m=CatBoostClassifier();m.load_model(str(paths(*key)[0]));models[key]=m
 return models
def process(a,fold,models):
 q,h=select(a);x=q.select(PC).to_numpy();hx=h.select(PC).to_numpy();p=reference(x,fold,models);ph=reference(hx,fold,models);prob,mass=calibrate(q,h,p,ph);out=q.select('table_id','hand_id','player_id','action_no','source_row','action_class','street_no').with_columns(pl.lit(fold).alias('fold'),*[pl.Series(f'{kind}_{k}',v[:,k]) for kind,v in [('reference',p),*prob.items()] for k in range(4)],pl.Series('context_count',mass[:,0]),pl.Series('kernel_mass',mass[:,1]));return out,q,h
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'tables').mkdir(exist_ok=True);cp=ROOT/'config.json'
 if cp.exists():assert json.loads(cp.read_text())==CONFIG
 else:cp.write_text(json.dumps(CONFIG,indent=2))
 models=load_models();tf=json.load(open('artifacts/policy/table_folds.json'));start=time.time();manifest={str(paths(*key)[0]):hashlib.file_digest(paths(*key)[0].open('rb'),'sha256').hexdigest() for key in models};(ROOT/'reference_hashes.json').write_text(json.dumps(manifest,indent=2));audit=[]
 for t,path in enumerate(sorted(Path('artifacts/policy/actions').glob('*.parquet'))):
  dest=ROOT/'tables'/path.name
  if not dest.exists():
   a=pl.read_parquet(path).filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');out,q,h=process(a,tf[path.stem],models);out.write_parquet(dest);audit.append({'table_id':path.stem,'queries':len(q),'context_actions':len(h),'shared_query_context_hands':0})
  if (t+1)%25==0:print('player kernels',t+1,round(time.time()-start,1),flush=True)
 (ROOT/'run_audit.json').write_text(json.dumps(audit,indent=2));compare()
def compare():
 q=pl.read_parquet(ROOT/'tables'/'*.parquet');y=q['action_class'].to_numpy().astype(int);loss={k:-np.log(q.select([f'{k}_{j}' for j in range(4)]).to_numpy()[np.arange(len(q)),y].clip(1e-7)) for k in CONFIG['arms']};z=q.with_columns(*[pl.Series(k,v) for k,v in loss.items()]);pool=z.group_by('table_id').agg(C(CONFIG['arms']).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(9412).integers(0,len(pool),(5000,len(pool)));report={'metric':'heldout ordinary-action logloss; not evidence MAP or Kaggle score','actions':len(q),'pools':len(pool),'mean_logloss':{k:float(v.mean()) for k,v in loss.items()},'folds':z.group_by('fold').agg(C(CONFIG['arms']).mean()).sort('fold').to_dicts(),'deltas':{}}
 for a,b in [('intercept','reference'),('kernel','reference'),('kernel','intercept')]:
  diff=pool[a].to_numpy()-pool[b].to_numpy();boot=diff[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report['deltas'][f'{a}_minus_{b}']={'mean':float((loss[a]-loss[b]).mean()),'pool_CI95':np.quantile(boot,[.025,.975]).tolist()}
 (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));pool.write_parquet(ROOT/'pool_losses.parquet');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
