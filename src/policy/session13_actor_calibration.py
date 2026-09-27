\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from scipy.optimize import minimize
from scipy.special import softmax
from catboost import CatBoostClassifier,CatBoostRegressor
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session13_actor');P=Path('artifacts/policy');C=pl.col
CONFIG={'action_ridge':.05,'size_ridge':.1,'max_iterations':50,'minimum_other_actions':50,'minimum_other_raises':20,'fit_exclusion':'all shared focal-pair hands, not merely evidence hands','features':'intercept/equity/street/pot odds/call-stack/log-stack/players-active/phase-relative time; partner-excluded normalization','size':'three Huber IRLS ridge steps on residual log bet ratio','audit':'time_index modulo five held out among non-partner hands for action prediction audit, followed by refit on all non-partner hands','arms':'same role/context aggregates from original vs adapted ordinary predictions'}

def design(a):
 return np.column_stack([a['equity'],a['street_no'].to_numpy()/3,a['pot_odds'],a['call_stack'],np.log1p(a['stack_bb'].to_numpy()),a['players_active'].to_numpy()/6,a['time_index'].to_numpy()/5000]).astype(float)
def standardize(z,tr):
 mu=z[tr].mean(0);sd=np.maximum(.1,z[tr].std(0));return np.column_stack([np.ones(len(z)),np.clip((z-mu)/sd,-6,6)]),mu,sd

def action_objective(b,x,lp,y,ridge):
 b=b.reshape(x.shape[1],4);v=lp+x@b;q=softmax(v,1);n=len(y);loss=-np.log(np.maximum(q[np.arange(n),y],1e-300)).mean()+ridge*.5*(b*b).sum();q[np.arange(n),y]-=1;gradient=x.T@q/n+ridge*b;return loss,gradient.ravel()
def fit_action(x,lp,y):
 init=np.zeros(x.shape[1]*4);res=minimize(action_objective,init,args=(x,lp,y,CONFIG['action_ridge']),jac=True,method='L-BFGS-B',options={'maxiter':50,'ftol':1e-10});zero=action_objective(init,x,lp,y,CONFIG['action_ridge'])[0];assert res.fun<=zero+1e-8;return res.x.reshape(x.shape[1],4),{'converged':bool(res.success),'iterations':int(res.nit),'objective_gain':float(zero-res.fun)}
def fit_size(x,r):
 b=np.zeros(x.shape[1]);w=np.ones(len(r))
 for _ in range(3):
  b=np.linalg.solve(x.T@(x*w[:,None])+np.eye(x.shape[1])*CONFIG['size_ridge']*len(x),x.T@(w*r));w=np.minimum(1,.5/np.maximum(abs(r-x@b),1e-8))
 return b

def process(table,query,models,sizes,cols):
 a=pl.read_parquet(P/'actions'/f'{table}.parquet').filter(C('phase')=='development').sort('time_index','action_no','player_id');f=json.load(open(P/'table_folds.json'))[table];x=a.select(cols).to_numpy();base=models[f].predict_proba(x,thread_count=2);size=sizes[f].predict(x,thread_count=2);call=a['to_call'].to_numpy()>0;legal=np.ones_like(base,bool);legal[call,1]=False;legal[~call,0]=False;legal[~call,2]=False;base*=legal;base/=base.sum(1)[:,None];y=a['action_class'].to_numpy().astype(int);assert legal[np.arange(len(y)),y].all();lp=np.where(legal,np.log(np.maximum(base,1e-30)),-1e9);Z=design(a);seat=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').select('hand_id','player_id','net_chips');states=pl.read_parquet(P/'states'/f'{table}.parquet').select('hand_id','player_id','street_no','fold_no');parts=[];audit=[]
 for (pid,),g in query.group_by('pair_id'):
  p1=g['player_1'][0];p2=g['player_2'][0];shared=set(g['hand_id']);target_hand=np.array([h in shared for h in a['hand_id']]);actorparts=[]
  for who,partner in [(p1,p2),(p2,p1)]:
   actor=a['player_id'].to_numpy()==who;off=actor&~target_hand;on=actor&target_hand;ix=np.flatnonzero(on)
   if not len(ix):continue
   calibrated=base[ix].copy();sz=size[ix].copy();info={'pair_id':pid,'actor':who,'other_actions':int(off.sum()),'shared_actions':len(ix),'shared_fit_overlap':int((off&on).sum())};assert not np.any(off&on)
   if off.sum()>=50:
    xx,mu,sd=standardize(Z,off);b,diagnostic=fit_action(xx[off],lp[off],y[off]);calibrated=softmax(lp[ix]+xx[ix]@b,1);info.update(diagnostic)
    held=off&(a['time_index'].to_numpy()%5==0);tr=off&~held
    if tr.sum()>=50 and held.sum()>=20:
     xxh,_,_=standardize(Z,tr);bh,_=fit_action(xxh[tr],lp[tr],y[tr]);pp=softmax(lp[held]+xxh[held]@bh,1);yy=y[held];info['audit_actions']=int(held.sum());info['base_audit_nll_sum']=float(-np.log(np.maximum(base[held][np.arange(len(yy)),yy],1e-30)).sum());info['adapted_audit_nll_sum']=float(-np.log(np.maximum(pp[np.arange(len(yy)),yy],1e-30)).sum())
    sr=off&(y==3)
    if sr.sum()>=20:
     bb=fit_size(xx[sr],a['log_bet_ratio'].to_numpy()[sr]-size[sr]);sz=size[ix]+xx[ix]@bb
   info['max_probability_sum_error']=float(abs(calibrated.sum(1)-1).max());assert info['max_probability_sum_error']<1e-12;assert np.max(calibrated[~legal[ix]],initial=0)==0;audit.append(info)
   z=a[ix].select('hand_id','player_id','street_no','action_no','action_class','last_aggressor','players_active','log_bet_ratio').with_columns(pl.lit(pid).alias('pair_id'),pl.lit(partner).alias('partner'),*[pl.Series(f'{arm}_p{k}',p[:,k]) for arm,p in [('global',base[ix]),('adapted',calibrated)] for k in range(4)],pl.Series('global_size_resid',a['log_bet_ratio'].to_numpy()[ix]-size[ix]),pl.Series('adapted_size_resid',a['log_bet_ratio'].to_numpy()[ix]-sz));actorparts.append(z)
  z=pl.concat(actorparts).with_columns(C('street_no').cast(pl.Int64)).join(seat.rename({'net_chips':'net'}),on=['hand_id','player_id']).join(seat.rename({'player_id':'partner','net_chips':'partner_net'}),on=['hand_id','partner']).join(states.rename({'player_id':'partner','fold_no':'partner_fold'}),on=['hand_id','partner','street_no']);alive=C('partner_fold')>=C('action_no');facing=(C('last_aggressor')==C('partner')).fill_null(False)&alive;contexts={'alive':alive,'partner':facing,'outside':alive&~facing&(C('players_active')>=3),'hu':alive&(C('players_active')==2)};expr=[]
  for arm in ['global','adapted']:
   for role,rmask in [('lower',C('net')<=C('partner_net')),('higher',C('net')>=C('partner_net'))]:
    for context,cmask in contexts.items():
     mask=rmask&cmask
     for k in range(4):
      p=C(f'{arm}_p{k}');event=C('action_class')==k;prefix=f'{arm}_{role}_{context}_{k}';expr.extend([((event.cast(pl.Float64)-p).filter(mask).sum()).alias(prefix+'_r'),(p*(1-p)).filter(mask).sum().alias(prefix+'_v'),(-p.clip(1e-30,1).log()).filter(mask&event).max().fill_null(0).alias(prefix+'_surprise'),p.filter(mask).mean().fill_null(0).alias(prefix+'_expected')])
     r=C(arm+'_size_resid');raise_mask=mask&(C('action_class')==3);prefix=f'{arm}_{role}_{context}_size';expr.extend([r.filter(raise_mask).sum().alias(prefix+'_sum'),r.filter(raise_mask).max().fill_null(0).alias(prefix+'_max'),r.filter(raise_mask).min().fill_null(0).alias(prefix+'_min'),r.abs().filter(raise_mask).max().fill_null(0).alias(prefix+'_absmax')])
  parts.append(z.group_by('pair_id','hand_id').agg(expr).with_columns(pl.selectors.numeric().cast(pl.Float32)))
 return pl.concat(parts),audit

def check():
 rng=np.random.default_rng(1313);x=rng.normal(size=(20,8));lp=np.log(rng.dirichlet([2,2,2,2],size=20));y=rng.integers(0,4,20);b=rng.normal(size=32)*.1;v,g=action_objective(b,x,lp,y,.05);err=0
 for k in [0,3,7,15,31]:
  e=np.zeros(32);e[k]=1e-6;diff=(action_objective(b+e,x,lp,y,.05)[0]-action_objective(b-e,x,lp,y,.05)[0])/2e-6;err=max(err,abs(diff-g[k]))
 assert err<1e-7;return {'action_objective_gradient_error':err}
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));(ROOT/'mathematical_checks.json').write_text(json.dumps(check(),indent=2));d=hand_data();players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');query=d.select('pair_id','hand_id','table_id').join(players,on='pair_id',validate='m:1');cols=json.load(open(P/'feature_columns.json'));models=[];sizes=[]
 for f in range(4):
  m=CatBoostClassifier();m.load_model(str(P/f'action_fold{f}.cbm'));models.append(m);s=CatBoostRegressor();s.load_model(str(P/f'size_fold{f}.cbm'));sizes.append(s)
 start=time.time();all_audit=[]
 for i,((table,),q) in enumerate(query.group_by('table_id')):
  path=ROOT/f'{table}.parquet'
  if path.exists():all_audit.extend(json.load(open(path.with_suffix('.json'))));continue
  out,audit=process(table,q.drop('table_id'),models,sizes,cols);assert set(out.select('pair_id','hand_id').iter_rows())==set(q.select('pair_id','hand_id').iter_rows());assert np.isfinite(out.select(pl.selectors.numeric()).to_numpy()).all();out.write_parquet(path,compression='zstd');path.with_suffix('.json').write_text(json.dumps(audit,indent=2));all_audit.extend(audit)
  if i%30==0:print('actor calibration',i,'seconds',round(time.time()-start,1),flush=True)
 pl.read_parquet(list(ROOT.glob('T*.parquet'))).write_parquet(ROOT/'hand_features.parquet');(ROOT/'fit_audit.json').write_text(json.dumps(all_audit,indent=2));n=sum(a.get('audit_actions',0) for a in all_audit);report={'actor_pair_fits':len(all_audit),'other_action_audit_rows':n,'base_other_action_logloss':sum(a.get('base_audit_nll_sum',0) for a in all_audit)/n,'adapted_other_action_logloss':sum(a.get('adapted_audit_nll_sum',0) for a in all_audit)/n,'shared_fit_overlap':sum(a['shared_fit_overlap'] for a in all_audit),'note':'Other-action audit is auxiliary self-supervision, not evidence MAP; repeated actors may appear for different partners'};(ROOT/'policy_audit.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
