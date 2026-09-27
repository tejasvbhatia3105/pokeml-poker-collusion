import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data
from session13_action_events import aggregate
from session13_action_residual import Model,bag_log_not
C=pl.col
def mathchecks():
 p=np.array([.12,.43,.71]);mass=0.;marg=np.zeros(3)
 for z in itertools.product(range(2),repeat=3):
  z=np.array(z);w=np.prod(np.where(z,p,1-p))
  if z.any():mass+=w;marg+=w*z
 err=max(abs(mass-(1-np.prod(1-p))),abs(marg/mass-p/mass).max());assert err<1e-14
 g=torch.tensor([0,0,1]);yy=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64)
 def fun(v):
  q=bag_log_not(v,g,2);return -(yy*(-torch.expm1(q)).log()+(1-yy)*q).sum()
 v=torch.tensor([[-2.,-3.],[-1.,-2.],[-2.,-1.]],dtype=torch.float64,requires_grad=True);assert torch.autograd.gradcheck(fun,(v,))
 return {'exhaustive_witness_posterior_error':err,'noisy_or_gradient_check':True}
def main():
 d=hand_data();reports={'mathematical':mathchecks()};actor=Path('artifacts/evidence_session13_actor');z=d.join(pl.read_parquet(actor/'hand_features.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
 with threadpool_limits(limits=2):
  for root in sorted(actor.iterdir()):
   if not root.is_dir() or not (root/'event_oof.parquet').exists():continue
   cols=json.load(open(root/'columns.json'));X=z.select(cols).to_numpy();hp='hist' in root.name;scores=pl.read_parquet(root/'event_oof.parquet');q=d.select('pair_id','hand_id','fold','behavior_family').join(scores.select('pair_id','hand_id',*(['new_hist_primary','new_hist_secondary'] if hp else ['bg_primary','bg_secondary'])),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.;n=0
   for f in range(4):
    for fam in ['directed_transfer','soft_play','coordinated_isolation']:
     use=((q['fold']==f)&(q['behavior_family']==fam)).to_numpy()
     for k,c in enumerate(['new_hist_primary','new_hist_secondary'] if hp else ['bg_primary','bg_secondary'],1):
      if hp:m=joblib.load(root/f'event{k}_{fam}_fold{f}.joblib');p=m.predict_proba(X[use])[:,1]
      else:m=CatBoostClassifier();m.load_model(str(root/f'event{k}_{fam}_fold{f}.cbm'));p=m.predict_proba(X[use],thread_count=2)[:,1]
      err=max(err,float(abs(p-q[c].to_numpy()[use]).max()));n+=1
   assert err<1e-12;reports[root.name]={'replayed_heads':n,'score_max_error':err,'features':len(cols)}
 a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet');cols=[c for c in a.columns if c not in ['pair_id','hand_id','bag_id','fold','evidence','behavior_family','time']];a=a.select('pair_id','hand_id',*cols).join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='m:1',maintain_order='left');X=a.select(cols).to_numpy().astype(np.float32);g=a['row'].to_numpy();cnt=np.bincount(g,minlength=len(d));fv=d['fold'].to_numpy();family=d['behavior_family'].to_numpy();root=Path('artifacts/evidence_session13_action_events');saved=d.select('pair_id','hand_id').join(pl.read_parquet(root/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.
 for f in range(4):
  for fam in ['directed_transfer','soft_play','coordinated_isolation']:
   va=(fv==f)&(family==fam);ix=np.flatnonzero(va[g])
   for k,c in enumerate(['bg_primary','bg_secondary'],1):
    m=CatBoostClassifier();m.load_model(str(root/f'action{k}_{fam}_fold{f}_em2.cbm'));p=m.predict_proba(X[ix],thread_count=2)[:,1];out=aggregate(p,g[ix],len(d));err=max(err,float(abs(out[va]-saved[c].to_numpy()[va]).max()))
 assert err<1e-12;reports['action_events']={'replayed_heads':24,'score_max_error':err}
 root=Path('artifacts/evidence_session13_action_residual')
 for kind in ['linear','nonlinear']:
  if not (root/kind/'event_oof.parquet').exists():continue
  saved=d.select('pair_id','hand_id').join(pl.read_parquet(root/kind/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.;normerr=0.
  for f in range(4):
   raw=d.select('pair_id','hand_id').join(pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');hp=raw.select('cat_primary','cat_secondary').to_numpy().clip(1e-7,1-1e-7);pa=-np.expm1(np.log1p(-hp[g])/cnt[g,None]);prior=np.log(pa)-np.log1p(-pa)
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    st=torch.load(root/kind/f'{fam}_fold{f}.pt',weights_only=False);va=(fv==f)&(family==fam);ix=np.flatnonzero(va[g]);tr=np.flatnonzero((fv[g]!=f)&(family[g]==fam));normerr=max(normerr,float(abs(X[tr].mean(0)-st['mu']).max()),float(abs(np.maximum(.05,X[tr].std(0))-st['sd']).max()));m=Model(len(cols),kind);m.load_state_dict(st['state_dict']);m.eval()
    with torch.no_grad():p=-torch.expm1(bag_log_not(torch.tensor(prior[ix],dtype=torch.float32)+m(torch.tensor(np.clip((X[ix]-st['mu'])/st['sd'],-6,6))),torch.tensor(g[ix].astype(np.int64)),len(d))).numpy()
    err=max(err,float(abs(p[va]-saved.select('bg_primary','bg_secondary').to_numpy()[va]).max()))
  assert err<1e-7 and normerr==0;reports['action_residual_'+kind]={'replayed_models':12,'score_max_error':err,'train_normalization_max_error':normerr}
 Path('artifacts/evidence_session13_verification.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
