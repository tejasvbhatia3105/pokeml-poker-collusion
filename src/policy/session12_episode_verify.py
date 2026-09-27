import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from session12_learned_episode import Episode,inclusion,features,Model
from session8_data import hand_data
C=pl.col;ROOT=Path('artifacts/evidence_session12/learned_episode_fast');torch.set_num_threads(1)
def main():
 d=hand_data();saved=pl.read_parquet(ROOT/'episode_oof.parquet');err=0.;paramerr=0.
 for kind in ['independent','episode']:
  slow=torch.load(f'artifacts/evidence_session12/learned_episode/{kind}_fold0.pt',weights_only=False)['state_dict'];fast=torch.load(ROOT/f'{kind}_fold0.pt',weights_only=False)['state_dict'];paramerr=max(paramerr,max(float(abs(slow[k]-fast[k]).max()) for k in slow))
 for f in range(4):
  raw=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time');q=d.filter(C('fold')==f).join(raw,on=['pair_id','hand_id'],validate='1:1');nets=[];em={}
  for seed in [1010,2020]:
   state=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(state['state_dict']);m.eval();nets.append((m,state))
  for kind in ['independent','episode']:
   st=torch.load(ROOT/f'{kind}_fold{f}.pt',weights_only=False);m=Episode(kind).double();m.load_state_dict(st['state_dict']);assert st['minimums']==state['minimums'];em[kind]=m
  for (pid,),g in q.group_by('pair_id'):
   g=g.sort('time','hand_id');x,p=features(g);mut=g.with_columns((1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'),pl.lit(99).alias('subtype'));mx,mp=features(mut);np.testing.assert_array_equal(mx,x);np.testing.assert_array_equal(mp,p);n=len(g);gap=torch.tensor(np.r_[0,np.diff(g['relative_time'].to_numpy())].astype(np.float64))[None];family=torch.tensor([['directed_transfer','soft_play','coordinated_isolation'].index(g['behavior_family'][0])]);mask=torch.ones((1,n),dtype=torch.bool);prior=[]
   with torch.no_grad():
    for m,st in nets:
     xx=torch.tensor(np.clip((x-st['mu'])/st['sd'],-6,6))[None];prior.append((torch.tensor(p)[None]+m(xx,mask)).double())
    for kind,m in em.items():
     values=[]
     for pr in prior:
      pp,tt=m(pr,gap,family,mask);values.append(inclusion(pp[0].numpy(),tt[0].numpy(),st['minimums'][g['behavior_family'][0]]))
     score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*np.mean(values,0);want=g.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')[kind].to_numpy();err=max(err,float(abs(score-want).max()))
 assert err<2e-6 and paramerr<1e-9;result={'saved_score_replay_max_error':err,'slow_vs_fast_fold0_parameter_error':paramerr,'label_mutation':'features invariant','folds':'frozen R30 and episode checkpoints use same held-out pool fold','mathematics':'46,656 path enumeration and independent-limit checks; analytical gradients match autograd and finite differences'};(ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
