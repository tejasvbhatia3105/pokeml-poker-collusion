\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from session8_data import hand_data
from session11_conditional_family import Model,features
from session8_count_conditioning import conditioned
ROOT=Path('artifacts/evidence_session13_pooling');C=pl.col;torch.set_num_threads(2)
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();parts=[];audit=[]
 for f in range(4):
  frames=[];models=[]
  for seed in [1010,2020]:
   st=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(st['state_dict']);models.append((st,m.eval()))
  for j in range(4):
   if j==f:continue
   a=json.load(open(f'artifacts/evidence_session9/nested_outer{j}_audit.json'))
   for r in a:
    if r['inner']==f:assert f not in r['training_folds'] and j not in r['training_folds'];audit.append(r)
   q=d.filter(C('fold')==f).join(pl.read_parquet(f'artifacts/evidence_session9/nested_outer{j}.parquet').filter(C('fold')==f).drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');frames.append({pid:g.sort('time','hand_id') for (pid,),g in q.group_by('pair_id')})
  for pid in frames[0]:
   probabilities=[];bases=[];cats=[]
   for fr in frames:
    g=fr[pid];x,p=features(g);bases.append(g['base'].to_numpy());cats.append(g['cat_inclusion'].to_numpy())
    with torch.no_grad():
     for st,m in models:
      delta=m(torch.tensor(np.clip((x-st['mu'])/st['sd'],-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];v=torch.tensor(p)+delta;probabilities.append(torch.softmax(torch.cat([torch.zeros_like(v[:,:1]),v],1),1).numpy().astype(float))
   k=st['minimums'][g['behavior_family'][0]];pp=np.stack(probabilities);linear=pp.mean(0);logs=np.log(np.maximum(pp,1e-12)).mean(0);geometric=np.exp(logs-logs.max(1)[:,None]);geometric/=geometric.sum(1)[:,None];fixed=.25*np.mean(bases,0)+.25*np.mean(cats,0);mix=fixed+.5*np.mean([conditioned(p[:,1:],k) for p in pp],0);lp=fixed+.5*conditioned(linear[:,1:],k);gp=fixed+.5*conditioned(geometric[:,1:],k);parts.append(g.select('pair_id','hand_id').with_columns(pl.Series('mixture',mix),pl.Series('linear_pool',lp),pl.Series('geometric_pool',gp)))
  print('pooling fold',f,flush=True)
 pl.concat(parts).write_parquet(ROOT/'pooling_oof.parquet');(ROOT/'audit.json').write_text(json.dumps({'teacher_exclusions':audit,'protocol':__doc__,'pooling':'six distributions: three smaller event teachers times two frozen correction seeds'},indent=2))
if __name__=='__main__':main()
