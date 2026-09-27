import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl,torch,joblib
import session89_crop_list_training as s
import session11_list_boost as original
C=pl.col
def inputs():
 d=s.data();original.CONFIG=dict(original.CONFIG,input_root=s.CONFIG['input_root']);rows=[];torch.set_num_threads(3)
 for f in range(4):
  full=s.pack(d,f,False);g,x,big,P,M,T,V,D,fv,bid,pos=original.pack(d,f,s.GROUND_COLUMNS);np.testing.assert_array_equal(full['X'],np.nan_to_num(np.column_stack([x,big]),nan=0,posinf=1e6,neginf=-1e6))
  for a,b in [('P',P),('M',M),('T',T),('V',V),('D',D)]:np.testing.assert_array_equal(full[a].numpy(),b.numpy())
  np.testing.assert_array_equal(full['fv'],fv);np.testing.assert_array_equal(full['bid'],bid);np.testing.assert_array_equal(full['pos'],pos);z=s.pack(d,f);mut=d.with_columns(pl.when(C('fold')==f).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when((C('fold')==f)&C('evidence_rank').is_not_null()).then(6-C('evidence_rank')).otherwise(C('evidence_rank')).alias('evidence_rank'));m=s.pack(mut,f);tr=z['tr'];tm=m['tr'];np.testing.assert_array_equal(z['X'][tr[z['bid']]],m['X'][tm[m['bid']]])
  for col in ['P','M','T','V','D','W','K']:np.testing.assert_array_equal(z[col].numpy()[tr],m[col].numpy()[tm])
  assert z['minimums']==m['minimums'];A=torch.zeros_like(z['P'],requires_grad=True);loss=s.objective(z,A);loss.backward();B=torch.zeros_like(m['P'],requires_grad=True);mloss=s.objective(m,B);mloss.backward();assert float(abs(loss-mloss).detach())==0;np.testing.assert_array_equal(A.grad.numpy()[tr],B.grad.numpy()[tm]);assert np.max(abs(A.grad.numpy()[~tr]))==0;total={};crop_views=0
  for (window,g),w in zip(z['groups'],z['W'].tolist()):
   pid=g['pair_id'][0];total[pid]=total.get(pid,0)+w
   if window!='full':assert set(g.filter(C('evidence')==1)['hand_id'])==set(d.filter((C('pair_id')==pid)&(C('evidence')==1))['hand_id']);crop_views+=1
  assert max(abs(v-1) for v in total.values())<1e-6;rows.append({'fold':f,'full_control_inputs_and_tensors_exact':True,'training_feature_target_weight_mutation_error':0,'training_loss_gradient_mutation_error':0,'heldout_gradient':0,'complete_list_crop_views':crop_views,'per_relationship_weight_error':max(abs(v-1) for v in total.values())})
 (s.ROOT/'input_verification.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
def models():
 d=s.data();saved=pl.read_parquet(s.ROOT/'oof.parquet');old=pl.read_parquet('artifacts/evidence_session62_grounded_list_boost/conditional_boost_oof.parquet').select('pair_id','hand_id','full');err=controlerr=0;views=0
 for f in range(4):
  z=s.pack(d,f);model=joblib.load(s.ROOT/f'list_boost_fold{f}.joblib');control=joblib.load(f'artifacts/evidence_session62_grounded_list_boost/list_boost_full_fold{f}.joblib')
  for i in np.flatnonzero(z['fv']==f):
   window,g=z['groups'][i];x=z['X'][z['bid']==i];p=z['P'].numpy()[i,:len(g)];new=s.scores(g,p,model,x);base=s.scores(g,p,control,x);expected=g.select('pair_id','hand_id').join(saved.filter(C('window')==window),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=max(err,float(abs(new-expected['tree_augmented'].to_numpy()).max()),float(abs(base-expected['tree_control'].to_numpy()).max()));views+=1
   if window=='full':q=g.select('pair_id','hand_id').join(old,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');controlerr=max(controlerr,float(abs(base-q['full'].to_numpy()).max()))
 assert err==controlerr==0;report={'models_replayed':4,'control_models_replayed':4,'views_replayed':views,'candidate_score_error':err,'original62_full_control_error':controlerr};(s.ROOT/'model_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='inputs':inputs()
 else:models()
