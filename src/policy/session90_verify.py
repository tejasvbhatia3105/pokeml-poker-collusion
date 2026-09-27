import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl,torch,joblib
import session90_partial_crop_list as s
from session11_conditional_family import template,list_nll,log_count_at_least
C=pl.col;FULL_PACK=s.engine.pack;FULL_OBJECTIVE=s.engine.objective
def enumeration():
 rng=np.random.default_rng(9012);p=rng.dirichlet([3,1,1],size=6);visible=np.array([True,False,True,False,True,False]);delta=rng.normal(0,.3,(6,2));logits=np.log(p[:,1:]/p[:,:1]);edited=logits+delta*visible[:,None];q=torch.softmax(torch.tensor(np.column_stack([np.zeros(6),edited])),1).numpy();found={};den=0
 for categories in itertools.product(range(3),repeat=6):
  mass=float(np.prod(q[np.arange(6),categories]));chosen=tuple(([i for i,k in enumerate(categories) if k==1]+[i for i,k in enumerate(categories) if k==2])[:5]);found[chosen]=found.get(chosen,0)+mass
  if sum(k!=0 for k in categories)>=3:den+=mass
 err=0;tested=0
 for e,expected in found.items():
  if len(e)<3:continue
  t=template(6,np.array(e));z=torch.tensor(edited)[None];nll=list_nll(z,torch.tensor(t)[None],torch.ones((1,len(t)),dtype=torch.bool),torch.tensor([len(e)]));count=log_count_at_least(z,torch.ones((1,6),dtype=torch.bool),torch.tensor([3]));actual=float(torch.exp(-nll*len(e)-count));err=max(err,abs(actual-expected/den));tested+=1
 assert err<1e-12;return {'enumerated_category_paths':3**6,'conditional_lists_checked':tested,'partial_view_probability_error':err,'omitted_hands_keep_fixed_probabilities':True}
def inputs():
 torch.set_num_threads(3);d=s.engine.data();proof=enumeration();rows=[]
 for arm in ['complete','all']:
  s.configure(arm)
  for f in range(4):
   z=s.pack(d,f);base=FULL_PACK(d,f,False);zero=torch.zeros_like(z['P']);initial=float(s.objective(z,zero));original=float(FULL_OBJECTIVE(base,torch.zeros_like(base['P'])));assert abs(initial-original)<1e-4;mut=d.with_columns(pl.when(C('fold')==f).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when((C('fold')==f)&C('evidence_rank').is_not_null()).then(6-C('evidence_rank')).otherwise(C('evidence_rank')).alias('evidence_rank'));m=s.pack(mut,f);tr=z['tr'];tm=m['tr'];np.testing.assert_array_equal(z['X'][tr[z['bid']]],m['X'][tm[m['bid']]])
   for col in ['P','M','F','T','V','D','W','K']:np.testing.assert_array_equal(z[col].numpy()[tr],m[col].numpy()[tm])
   A=zero.requires_grad_(True);loss=s.objective(z,A);loss.backward();B=torch.zeros_like(m['P'],requires_grad=True);mloss=s.objective(m,B);mloss.backward();assert float(abs(loss-mloss).detach())==0;np.testing.assert_array_equal(A.grad.numpy()[tr],B.grad.numpy()[tm]);assert abs(A.grad.numpy()[~tr]).max()==0;assert abs(A.grad.numpy()[~z['M'].numpy()]).max()==0
   for i,(window,g) in enumerate(z['groups']):
    pos=z['pos'][z['bid']==i];np.testing.assert_array_equal(pos,g['full_position'].to_numpy());assert int(z['M'][i].sum())==len(g);assert int(z['F'][i].sum())==len(z['originals'][i]);assert z['D'][i]==z['originals'][i]['evidence_rank'].is_not_null().sum()
   rows.append({'arm':arm,'fold':f,'views':len(z['groups']),'scorable_complete_views':int(z['predict_views'].sum()),'zero_correction_loss_difference_vs_original_full':initial-original,'training_input_target_gradient_mutation_error':0,'heldout_gradient':0,'omitted_hand_correction_gradient':0,'full_position_maps_exact':True})
 report={'enumeration':proof,'folds':rows};(s.ROOT/'input_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
def models():
 d=s.engine.data();rows=[];old=pl.read_parquet('artifacts/evidence_session62_grounded_list_boost/conditional_boost_oof.parquet').select('pair_id','hand_id','full')
 for arm in ['complete','all']:
  s.configure(arm);saved=pl.read_parquet(s.engine.ROOT/'oof.parquet');err=ctrl=0;views=0
  for f in range(4):
   z=s.pack(d,f);model=joblib.load(s.engine.ROOT/f'list_boost_fold{f}.joblib');control=joblib.load(f'artifacts/evidence_session62_grounded_list_boost/list_boost_full_fold{f}.joblib')
   for i in np.flatnonzero((z['fv']==f)&z['predict_views']):
    w,g=z['groups'][i];mask=z['bid']==i;x=z['X'][mask];prior=z['P'].numpy()[i,z['pos'][mask]];p=s.engine.scores(g,prior,model,x);b=s.engine.scores(g,prior,control,x);q=g.select('pair_id','hand_id').join(saved.filter(C('window')==w),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=max(err,float(abs(p-q['tree_augmented'].to_numpy()).max()),float(abs(b-q['tree_control'].to_numpy()).max()));views+=1
    if w=='full':r=g.select('pair_id','hand_id').join(old,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');ctrl=max(ctrl,float(abs(b-r['full'].to_numpy()).max()))
  assert err==ctrl==0;rows.append({'arm':arm,'models_replayed':4,'control_models_replayed':4,'views_replayed':views,'score_error':err,'original62_full_control_error':ctrl})
 (s.ROOT/'model_verification.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='inputs':inputs()
 else:models()
