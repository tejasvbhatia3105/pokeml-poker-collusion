import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl,torch,joblib
import session93_joint_transfer_list as s
from session12_compare import compare
C=pl.col
def exact():
 rng=np.random.default_rng(9301);p=rng.dirichlet([3,1,1],size=(2,6));found={};mass=0
 for actor in [0,1]:
  for cat in itertools.product(range(3),repeat=6):
   e=tuple(([i for i,k in enumerate(cat) if k==1]+[i for i,k in enumerate(cat) if k==2])[:5]);prob=.5*np.prod(p[actor,np.arange(6),cat]);found[e]=found.get(e,0)+prob
   if len(e)>=3:mass+=prob
 z=torch.tensor(np.log(p[:,:,1:]/p[:,:,:1]))[None];maximum=0;count=0
 for e,prob in found.items():
  if len(e)<3:continue
  t=s.template(6,np.array(e));value=s.list_loss(z,torch.tensor(t)[None],torch.ones((1,len(t)),dtype=torch.bool),torch.ones((1,6),dtype=torch.bool),torch.tensor([len(e)]),torch.tensor([3]));actual=np.exp(-value.item()*len(e));maximum=max(maximum,abs(actual-prob/mass));count+=1
 assert maximum<1e-12
                                                                              
 ap=np.array([.2,.4,.3,.1]);idx=torch.tensor([0,0,1,1]);zz=s.hand_logits(torch.tensor(np.log(ap/(1-ap))),idx,1,1);actual=torch.softmax(torch.cat([torch.zeros_like(zz[:,:,:,:1]),zz],3),3).numpy()[0,0,0];expect=np.zeros(3)
 for c in itertools.product([0,1],repeat=4):expect[1 if any(c[:2]) else 2 if any(c[2:]) else 0]+=np.prod(np.where(c,ap,1-ap))
 np.testing.assert_allclose(actual,expect,atol=1e-12)
 t=s.template(6,np.array([1,3,4]));z.requires_grad_(True);assert torch.autograd.gradcheck(lambda a:s.list_loss(a,torch.tensor(t)[None],torch.ones((1,len(t)),dtype=torch.bool),torch.ones((1,6),dtype=torch.bool),torch.tensor([3]),torch.tensor([3])),(z,),atol=1e-5)
 result={'enumerated_donor_category_paths':2*3**6,'eligible_lists_checked':count,'conditional_probability_error':maximum,'action_aggregation_error':float(abs(actual-expect).max()),'finite_difference_gradient_check':True};(s.ROOT/'exact_check.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
def verify():
 d,a,x,groups,T,V,M,D,fv,rb,head,indices=s.pack();saved=pl.read_parquet(s.ROOT/'direct_oof.parquet');records=[];rr=torch.tensor(rb)
 N,n=M.shape;support=np.bincount(indices.numpy(),minlength=N*2*n*2).reshape(N,2,n,2)>0;support_records=[]
 for i,q in enumerate(groups):
  opts=[]
  for actor in [0,1]:
   for j in np.flatnonzero(V[i].numpy()):
    t=T[i,j].numpy()
    if support[i,actor,t==1,0].all() and support[i,actor,t==2,1].all():opts.append([actor,int(j)])
  support_records.append({'pair_id':q['pair_id'][0],'fold':int(fv[i]),'options':opts,'compatible':bool(opts)})
 support_report={'pairs':N,'actions':len(a),'fields':x.shape[1],'compatible_lists':sum(r['compatible'] for r in support_records),'incompatible':[r for r in support_records if not r['compatible']],'records':support_records};(s.ROOT/'support_audit.json').write_text(json.dumps(support_report,indent=2))
 for f in range(4):
  m=joblib.load(s.ROOT/f'joint_fold{f}.joblib');A=torch.tensor(s.infer(m,x,head),requires_grad=True);np.testing.assert_array_equal(A.detach().numpy(),s.infer(m,x[::-1],head[::-1])[::-1]);train=torch.tensor(np.flatnonzero(fv!=f));rt=torch.tensor(fv[rb]!=f);K=torch.full((len(M),),m['minimum']);loss=s.objective(A,indices,T,V,M,D,train,K,rr,rt);g=torch.autograd.grad(loss,A)[0];TT=T.clone();VV=V.clone();DD=D.clone();TT[fv==f]=0;VV[fv==f]=False;DD[fv==f]=999;AA=A.detach().clone().requires_grad_(True);other=s.objective(AA,indices,TT,VV,M,DD,train,K,rr,rt);gg=torch.autograd.grad(other,AA)[0];assert float(loss.detach())==float(other.detach());np.testing.assert_array_equal(g.numpy(),gg.numpy());assert not torch.count_nonzero(g[~rt]);out=s.predict(m,x,head,indices,groups,M,fv);q=out.join(saved,on=['pair_id','hand_id'],validate='1:1',suffix='_saved')
  for c in ['list_inclusion','raw_inclusion']:np.testing.assert_array_equal(q[c].to_numpy(),q[c+'_saved'].to_numpy())
  records.append({'fold':f,'heldout_target_loss_gradient_change':0,'heldout_gradient':0,'model_replay_error':0,'query_permutation_error':0})
 report={'models_replayed':4,'paired_actions':len(a),'feature_fields':x.shape[1],'folds':records,'inference_donor':'preserved session25 out-of-fold probabilities; not used as training targets'};(s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
def comparison():
 base=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33'),'pressure59','full');raw=pl.concat([pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).select('pair_id','hand_id','base') for f in range(4)]);new=pl.read_parquet(s.ROOT/'direct_oof.parquet');out=base.join(new,on=['pair_id','hand_id'],how='left',validate='1:1').join(raw,on=['pair_id','hand_id'],validate='1:1').with_columns(pl.when(C('list_inclusion').is_not_null()).then(.25*C('base')+.25*C('raw_inclusion')+.5*C('list_inclusion')).otherwise(C('pressure59')).alias('candidate'));out=out.with_columns(((C('candidate')+C('full'))*.5).alias('candidate_r33_recipe'));out.write_parquet(s.ROOT/'oof.parquet');names=['r33','candidate','candidate_r33_recipe'];r,_=compare(s.ROOT/'oof.parquet',names,'session93');pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));report={}
 for k in names:
  delta=pool[k].to_numpy()-pool['r33'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report[k]={'MAP':r[k].mean(),'delta':r[k].mean()-r['r33'].mean(),'fixed_prediction_pool_CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(k).mean()).sort('fold')[k].to_list(),'families':dict(r.group_by('family').agg(C(k).mean()).iter_rows()),'better':int((r[k]>r['r33']+1e-12).sum()),'worse':int((r[k]<r['r33']-1e-12).sum())}
 (s.ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 import sys
 if len(sys.argv)>1 and sys.argv[1]=='exact':exact()
 else:exact();verify();comparison()
