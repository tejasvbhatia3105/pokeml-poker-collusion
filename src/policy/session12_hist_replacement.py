import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from session8_data import hand_data
from session11_conditional_family import Model,features
from session6_priority import inclusion
from session8_count_conditioning import conditioned
ROOT=Path('artifacts/evidence_session12/hist_pseudo');C=pl.col;torch.set_num_threads(2)
def assemble(root=ROOT,catpath=Path('artifacts/evidence_session12/unlabelled_gated/event_oof.parquet')):
 d=hand_data();hp=pl.read_parquet(root/'event_oof.parquet').select('pair_id','hand_id','new_hist_primary','new_hist_secondary');cp=pl.read_parquet(catpath).select('pair_id','hand_id','bg_primary','bg_secondary');parts=[]
 for f in range(4):
  raw=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time');q=d.filter(C('fold')==f).join(raw,on=['pair_id','hand_id'],validate='1:1').join(hp,on=['pair_id','hand_id'],validate='1:1').join(cp,on=['pair_id','hand_id'],validate='1:1');models=[]
  for seed in [1010,2020]:
   st=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(st['state_dict']);m.eval();models.append((st,m))
  for (pid,),g in q.group_by('pair_id'):
   g=g.sort('time','hand_id');out=g.select('pair_id','hand_id')
   for arm in ['hist','both']:
    ca=g.select('bg_primary','bg_secondary').to_numpy() if arm=='both' else g.select('cat_primary','cat_secondary').to_numpy();ha=g.select('new_hist_primary','new_hist_secondary').to_numpy();nc=ca/np.maximum(1,ca.sum(1))[:,None];nh=ha/np.maximum(1,ha.sum(1))[:,None];jp=.5*(nc+nh);ci=inclusion(nc[:,0],nc[:,1]);ji=inclusion(jp[:,0],jp[:,1]);newg=g.with_columns(pl.Series('cat_primary',ca[:,0]),pl.Series('cat_secondary',ca[:,1]),pl.Series('hist_primary',ha[:,0]),pl.Series('hist_secondary',ha[:,1]),pl.Series('cat_inclusion',ci),pl.Series('joint_inclusion',ji),pl.Series('r29',.25*g['base'].to_numpy()+.25*ci+.5*ji));nx,prior=features(newg);ox,_=features(g)
    for mode in ['prior','propagated']:
     inc=[];x=ox if mode=='prior' else nx
     with torch.no_grad():
      for st,m in models:
       delta=m(torch.tensor(np.clip((x-st['mu'])/st['sd'],-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];logits=torch.tensor(prior)+delta;p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:1]),logits],1),1).numpy();inc.append(conditioned(p[:,1:],st['minimums'][g['behavior_family'][0]]))
     score=.25*g['base'].to_numpy()+.25*ci+.5*np.mean(inc,0);out=out.with_columns(pl.Series(arm+'_'+mode,score))
   parts.append(out)
 pl.concat(parts).write_parquet(root/'hist_pseudo_oof.parquet')
if __name__=='__main__':assemble()
