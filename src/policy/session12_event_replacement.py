import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import polars as pl,numpy as np,torch
from session8_data import hand_data
from session11_conditional_family import Model,features
from session6_priority import inclusion
from session8_count_conditioning import conditioned
C=pl.col
def assemble(ROOT):
 d=hand_data();q=pl.read_parquet(ROOT/'event_oof.parquet');r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet');parts=[]
 for f in range(4):
  raw=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time');z=d.filter(C('fold')==f).join(raw,on=['pair_id','hand_id'],validate='1:1').join(q.select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1').join(r30.select('pair_id','hand_id','conditional_family'),on=['pair_id','hand_id'],validate='1:1');models=[]
  for seed in [1010,2020]:
   st=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(st['state_dict']);m.eval();models.append((st,m))
  for (pid,),g in z.group_by('pair_id'):
   g=g.sort('time','hand_id');ca=g.select('bg_primary','bg_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];ci=inclusion(ca[:,0],ca[:,1]);base=g['conditional_family'].to_numpy();replace=base+.25*(ci-g['cat_inclusion'].to_numpy());hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];jp=.5*(ca+hp);prior=np.log(np.maximum(jp,1e-6))-np.log(np.maximum(1-jp.sum(1),1e-6))[:,None];x,_=features(g);incs=[]
   with torch.no_grad():
    for st,m in models:
     delta=m(torch.tensor(np.clip((x-st['mu'])/st['sd'],-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];logits=torch.tensor(prior,dtype=torch.float32)+delta;p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:1]),logits],1),1).numpy();incs.append(conditioned(p[:,1:],st['minimums'][g['behavior_family'][0]]))
   allreplace=.25*g['base'].to_numpy()+.25*ci+.5*np.mean(incs,0)
   ji=inclusion(jp[:,0],jp[:,1]);newg=g.with_columns(C('bg_primary').alias('cat_primary'),C('bg_secondary').alias('cat_secondary'),pl.Series('cat_inclusion',ci),pl.Series('joint_inclusion',ji),pl.Series('r29',.25*g['base'].to_numpy()+.25*ci+.5*ji));nx,np0=features(newg);prop=[]
   with torch.no_grad():
    for st,m in models:
     delta=m(torch.tensor(np.clip((nx-st['mu'])/st['sd'],-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];logits=torch.tensor(np0)+delta;p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:1]),logits],1),1).numpy();prop.append(conditioned(p[:,1:],st['minimums'][g['behavior_family'][0]]))
   propagated=.25*g['base'].to_numpy()+.25*ci+.5*np.mean(prop,0);parts.append(g.select('pair_id','hand_id').with_columns(pl.Series('cat_only',replace),pl.Series('cat_and_joint',allreplace),pl.Series('propagated',propagated)))
 pl.concat(parts).write_parquet(ROOT/'background_oof.parquet')
