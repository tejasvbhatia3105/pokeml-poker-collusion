import json
import numpy as np,polars as pl,torch
import session62_grounded_list_boost as task
from session8_count_conditioning import conditioned
def main():
 task.setup();d=task.ORIGINAL_DATA();px,cols=task.grounded(d);np.testing.assert_array_equal(px,np.load(task.ROOT/'grounded_features.npz')['x']);mutated=d.with_columns((1-pl.col('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'),pl.lit(99).alias('subtype'));mx,mc=task.grounded(mutated);np.testing.assert_array_equal(px,mx);assert cols==mc;task.PCOLS=cols;d=d.with_columns(*[pl.Series(c,px[:,i]) for i,c in enumerate(cols)]);saved=pl.read_parquet(task.ROOT/'conditional_boost_oof.parquet');error=permerr=0;models=0
 for f in range(4):
  groups,x,big,P,M,T,V,D,fv,bid,pos=task.pack(d,f,None);train=torch.tensor(np.flatnonzero((fv!=f)&V.any(1).numpy()));valid=np.flatnonzero(fv==f);va=np.isin(bid,valid);A=torch.zeros_like(P,requires_grad=True);loss=task.objective(P,A,M,T,V,D,train);loss.backward();assert torch.count_nonzero(A.grad[valid])==0;changed=T.clone();changed[valid]=1;den=D.clone();den[valid]=999;assert float(loss.detach())==float(task.objective(P,A.detach(),M,changed,V,den,train))
  for kind in ['compact','full']:
   X=x if kind=='compact' else np.column_stack([x,big]);X=np.nan_to_num(X,nan=0,posinf=1e6,neginf=-1e6);model=task.b.joblib.load(task.ROOT/f'list_boost_{kind}_fold{f}.joblib');assert model['feature_count']==X.shape[1];delta=task.b.infer(model,X);perm=np.random.default_rng(f).permutation(int(va.sum()));permerr=max(permerr,float(abs(task.b.infer(model,X[va][perm])-delta[va][perm]).max()));z=P.numpy().copy();z[bid,pos]+=delta;models+=1
   for i in valid:
    g=groups[i];n=len(g);p=torch.softmax(torch.tensor(np.column_stack([np.zeros(n,np.float32),z[i,:n]])),1).numpy();inc=conditioned(p[:,1:],model['minimums'][g['behavior_family'][0]]);score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;expected=g.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],maintain_order='left',validate='1:1');error=max(error,float(abs(score-expected[kind].to_numpy()).max()),float(abs(inc-expected[kind+'_inclusion'].to_numpy()).max()))
 assert error==0 and permerr==0;audit=json.load(open(task.ROOT/'list_boost_audit.json'))
 for r in audit:assert np.max(np.diff([r['initial_loss']]+r['training_loss']))<1e-5
 out={'models_replayed':models,'prediction_error':error,'row_permutation_error':permerr,'grounded_features_replay_exact':True,'grounded_features_label_rank_subtype_invariant':True,'heldout_objective_gradient_zero':True,'heldout_template_and_count_mutation_invariant':True,'training_line_search_monotone':True,'nested_teacher_input_root':task.b.CONFIG['input_root']};(task.ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
