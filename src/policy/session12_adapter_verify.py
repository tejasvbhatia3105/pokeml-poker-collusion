import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session12_frozen_adapter import ROOT,CONFIG,Model,features,template
from session8_data import hand_data
from session8_count_conditioning import conditioned
C=pl.col;torch.set_num_threads(2)
def design(g):
 x,p=features(g);v=g.select([f'emb_{i}' for i in range(128)]).to_numpy();f=['directed_transfer','soft_play','coordinated_isolation'].index(g['behavior_family'][0]);e=np.zeros((len(g),384),np.float32);e[:,f*128:(f+1)*128]=v;return x,p,np.concatenate([x,e],1)
def main():
 d=hand_data();saved=pl.read_parquet(ROOT/'adapter_oof.parquet');err=0.;initial=0.;models=0
 for f in range(4):
  q=d.join(pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet(f'artifacts/evidence_session12/action_embeddings/trained_outer{f}.parquet'),on=['pair_id','hand_id'],validate='1:1');bags=[]
  for (pid,),g in q.group_by('pair_id'):
   g=g.sort('time','hand_id');x,p,a=design(g);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];bags.append((pid,g,x,p,a,len(template(len(g),e))>0))
  bags.sort(key=lambda b:b[0]);train=[b for b in bags if b[1]['fold'][0]!=f and b[-1]];valid=[b for b in bags if b[1]['fold'][0]==f]
  for kind in ['compact','action']:
   mat=np.concatenate([b[2 if kind=='compact' else 4] for b in train]);mu=mat.mean(0);sd=np.maximum(.05,mat.std(0));nets=[]
   for seed in CONFIG['seeds']:
    st=torch.load(ROOT/f'adapter_{kind}_fold{f}_seed{seed}.pt',weights_only=False);assert st['config']==CONFIG;np.testing.assert_array_equal(mu,st['mu']);np.testing.assert_array_equal(sd,st['sd']);a=nn.Linear(len(mu),2);a.load_state_dict(st['state_dict']);base=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(base['state_dict']);m.eval();assert st['minimums']==base['minimums'];nets.append((a,m,base,st));models+=1
   for pid,g,x,p,a,_ in valid:
    mut=g.with_columns((1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'),pl.lit(99).alias('subtype'));mx,mp,ma=design(mut);np.testing.assert_array_equal(mx,x);np.testing.assert_array_equal(mp,p);np.testing.assert_array_equal(ma,a);inc=[]
    with torch.no_grad():
     for net,m,base,st in nets:
      mm=torch.ones((1,len(g)),dtype=torch.bool);xx=torch.tensor(np.clip((x-base['mu'])/base['sd'],-6,6))[None];prior=torch.tensor(p)+m(xx,mm)[0];aa=torch.tensor(np.clip(((x if kind=='compact' else a)-mu)/sd,-6,6));logits=prior+CONFIG['bound']*torch.tanh(net(aa));prob=torch.softmax(torch.cat([torch.zeros_like(logits[:,:1]),logits],1),1).numpy();inc.append(conditioned(prob[:,1:],st['minimums'][g['behavior_family'][0]]))
    score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*np.mean(inc,0);want=g.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')[kind].to_numpy();err=max(err,float(abs(score-want).max()))
 assert err<2e-6;result={'models':models,'saved_replay_max_error':err,'normalization':'exact outer-training replay','feature_label_mutation':'invariant','minimums':'match frozen R30 training-only minimums','decision':'Both arms failed to improve R30 and are not candidates'};(ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
