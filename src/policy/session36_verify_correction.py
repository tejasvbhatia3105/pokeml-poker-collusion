import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl,torch
from session36_nested_cascade import ROOT
from session8_data import hand_data
from session11_conditional_family import Model,features,template,CONFIG
from session8_count_conditioning import conditioned
C=pl.col
def main(root=ROOT,nested_kind='cascade'):
 ROOT=root;inp=json.load(open(ROOT/'input_verification.json'))
 if nested_kind=='cascade':assert inp['conditional_heads_replayed']==72 and inp['max_probability_error']==0
 else:assert nested_kind=='current' and not inp['pilot_only'] and inp['saved_models_replayed']==175 and inp['event_probability_error']==0 and inp['all_training_prediction_and_outer_pool_sets_disjoint']
 folder=ROOT/'correction';saved=pl.read_parquet(folder/'list_learning_oof.parquet');d=hand_data();maxerr=0.;permerr=0.;models=0
 for f in range(4):
  q=d.join(pl.read_parquet(ROOT/f'nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');bags=[]
  for (pid,),g in q.group_by('pair_id'):
   g=g.sort('time','hand_id');x,p=features(g);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];bags.append({'pid':pid,'g':g,'x':x,'p':p,'fold':g['fold'][0],'compatible':len(template(len(g),e))>0})
  bags.sort(key=lambda b:b['pid']);n=max(len(b['g']) for b in bags);N=len(bags);X=np.zeros((N,n,35),np.float32);P=np.zeros((N,n,2),np.float32);M=np.zeros((N,n),bool)
  for i,b in enumerate(bags):k=len(b['g']);X[i,:k]=b['x'];P[i,:k]=b['p'];M[i,:k]=True
  tr=np.array([b['fold']!=f and b['compatible'] for b in bags]);va=np.array([b['fold']==f for b in bags]);assert not (tr&va).any();mu=X[tr][M[tr]].mean(0);sd=np.maximum(.05,X[tr][M[tr]].std(0));xx=torch.tensor(np.clip((X[va]-mu)/sd,-6,6));pp=torch.tensor(P[va]);mm=torch.tensor(M[va]);pred=[];minimums={fam:min(int(b['g']['evidence'].sum()) for b in bags if b['fold']!=f and b['compatible'] and b['g']['behavior_family'][0]==fam) for fam in ['directed_transfer','soft_play','coordinated_isolation']}
  for seed in CONFIG['seeds']:
   st=torch.load(folder/f'list_independent_fold{f}_seed{seed}.pt',weights_only=False);assert st['config']==CONFIG and st['minimums']==minimums;np.testing.assert_array_equal(st['mu'],mu);np.testing.assert_array_equal(st['sd'],sd);m=Model(35,'independent');m.load_state_dict(st['state_dict']);m.eval()
   with torch.no_grad():
    delta=m(xx,mm);permerr=max(permerr,float(abs(delta-m(xx.flip(1),mm.flip(1)).flip(1)).max()));logits=pp+delta;pred.append(torch.softmax(torch.cat([torch.zeros_like(logits[:,:,:1]),logits],2),2).numpy());models+=1
  for j,b in enumerate([b for b in bags if b['fold']==f]):
   g=b['g'];k=len(g);inc=np.mean([conditioned(p[j,:k,1:3],minimums[g['behavior_family'][0]]) for p in pred],axis=0);score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;expected=g.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['independent'].to_numpy();maxerr=max(maxerr,float(abs(score-expected).max()));mut=g.with_columns((1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'),pl.lit(99).alias('subtype'));mx,mp=features(mut);np.testing.assert_array_equal(mx,b['x']);np.testing.assert_array_equal(mp,b['p'])
 assert maxerr<1e-6 and permerr<3e-6;out={'correction_models_replayed':models,'saved_prediction_error':maxerr,'permutation_error':permerr,'train_only_normalization_and_family_minimums_exact':True,'heldout_label_mutation_invariant':True,'nested_inputs':inp};(folder/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
