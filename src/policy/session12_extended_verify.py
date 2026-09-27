import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from threadpoolctl import threadpool_limits
from session8_data import hand_data,targets
C=pl.col

def type_math():
 from session12_type_mixture import TypeMixture,objective,template
 from session8_count_conditioning import conditioned
 rng=np.random.default_rng(121313);n=6;base=rng.dirichlet([3,1,1],size=n);m=TypeMixture()
 with torch.no_grad():m.gate.bias.fill_(.3);m.bias.fill_(.2)
 p,pi=m(torch.tensor(base)[None],torch.tensor(rng.normal(size=(1,21))),torch.tensor([1]));p=p.detach().numpy()[0];pi=pi.detach().numpy()[0];maxerr=0.;incerr=0.;checked=0
 for kind in ['independent','shared']:
  records={};inc={k:np.zeros(n) for k in [3,5]};mass={3:0.,5:0.}
  for cats in itertools.product(range(3),repeat=n):
   q=(p*pi[None,:,None]).sum(1);prob=np.prod(q[np.arange(n),cats]) if kind=='independent' else sum(pi[s]*np.prod(p[np.arange(n),s,cats]) for s in range(2));e=tuple(([i for i,c in enumerate(cats) if c==1]+[i for i,c in enumerate(cats) if c==2])[:5]);records[e]=records.get(e,0)+prob
   for k in mass:
    if len(e)>=k:mass[k]+=prob;inc[k][list(e)]+=prob
  for k in mass:
   actual=conditioned((p*pi[None,:,None]).sum(1)[:,1:],k) if kind=='independent' else sum(pi[s]*conditioned(p[:,s,1:],k) for s in range(2));incerr=max(incerr,float(abs(actual-inc[k]/mass[k]).max()))
   for e,pr in records.items():
    if len(e)<k:continue
    z=template(n,np.array(e));tt=np.zeros((1,6,n),np.int64);vv=np.zeros((1,6),bool);tt[0,:len(z)]=z;vv[0,:len(z)]=True;loss=objective(torch.tensor(p)[None],torch.tensor(pi)[None],torch.tensor(tt),torch.tensor(vv),torch.tensor([len(e)]),torch.tensor([np.log(mass[k])]),kind);err=abs(np.exp(-loss.item()*len(e))-pr/mass[k]);maxerr=max(maxerr,err);checked+=1
 assert max(maxerr,incerr)<1e-11;result={'observed_lists_checked':checked,'likelihood_error':maxerr,'inclusion_error':incerr,'event_rate_invariance':float(abs(p[:,:,1:].sum(2)-base[:,1:].sum(1)[:,None]).max())};Path('artifacts/evidence_session12/type_mixture/mathematical_checks.json').write_text(json.dumps(result,indent=2));print('type math',result,flush=True)

def hist_replay(d):
 root=Path('artifacts/evidence_session12/hist_pseudo');cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();saved=pl.read_parquet(root/'event_oof.parquet');err=0.
 with threadpool_limits(limits=2):
  for f in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    va=(d['fold'].to_numpy()==f)&(d['behavior_family'].to_numpy()==fam);g=d.filter(pl.Series(va)).select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    for k in [1,2]:
     m=joblib.load(root/f'event{k}_{fam}_fold{f}.joblib');p=m.predict_proba(X[va])[:,1];err=max(err,float(abs(p-g['new_hist_primary' if k==1 else 'new_hist_secondary'].to_numpy()).max()))
 assert err<1e-12;(root/'verification.json').write_text(json.dumps({'models_replayed':24,'event_score_replay_max_error':err,'pseudo_provenance':'../unlabelled_pseudo/provenance_verification.json'},indent=2));print('hist replay',err,flush=True)

def transductive(d):
 root=Path('artifacts/evidence_session12/transductive');folds=json.load(open('artifacts/policy/table_folds.json'));cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];err=0.;rows=[]
 for f in range(4):
  audit=[json.load(open(p)) for p in (root/f'outer{f}').glob('T*.json')];assert {r['table_id'] for r in audit}=={t for t,v in folds.items() if v==f};q=pl.read_parquet(list((root/f'outer{f}').glob('T*.parquet')));assert (q['outer_fold']==f).all();assert not set(q.columns)&{'evidence','evidence_rank','subtype','label'};assert q.select('pair_id','hand_id').n_unique()==len(q)
  z=q.join(d.select('pair_id','hand_id',*[C(c).cast(pl.Float32) for c in cols]),on=['pair_id','hand_id'],suffix='_reference',validate='1:1');local=max(float((z[c]-z[c+'_reference']).abs().max()) for c in cols);err=max(err,local)
  for name in ['primary','secondary']:
   p=q.filter(C('eligible_'+name))['pseudo_'+name];assert ((p<.02)|(p>.9)).all()
  rows.append({'fold':f,'candidate_pairs_scanned':sum(a['all_candidate_pairs'] for a in audit),'pseudo_pairs':q['pair_id'].n_unique(),'pseudo_hands':len(q),'public_positive_hands_checked_after_selection':len(z),'legacy_feature_max_error':local})
 assert err<1e-4;result={'protocol':'transductive: held-out gameplay used but real labels excluded; not inductive OOF','folds':rows,'selection':'all development candidate pairs scored with fixed held-out pair teacher; no true-label filtering','features':'exact legacy float32 comparison after pseudo selection','outer_targets':'original event target masks exclude outer fold; pseudo targets depend only on model probabilities'};(root/'provenance_verification.json').write_text(json.dumps(result,indent=2));print('transductive provenance',result,flush=True)
if __name__=='__main__':
 d=hand_data();type_math();hist_replay(d);transductive(d)
