\
\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from session57_isolation_pressure import data
from session67_isolation_types import design
from session11_conditional_family import Model,features
from session6_priority import inclusion
from session8_count_conditioning import conditioned
from session12_compare import compare
ROOT=Path('artifacts/evidence_session71_supported_list');C=pl.col
def project(p,support):
 p=np.asarray(p,float)*support;return p/np.maximum(1,p.sum(1))[:,None]
def score(g,models,support):
 ca=project(g.select('bg_primary','bg_secondary').to_numpy(),support);hp=project(g.select('hist_primary','hist_secondary').to_numpy(),support);jp=(ca+hp)/2;prior=np.log(np.maximum(jp,1e-6))-np.log(np.maximum(1-jp.sum(1),1e-6))[:,None];x,_=features(g);incs=[]
 for st,m in models:
  with torch.no_grad():delta=m(torch.tensor(np.clip((x-st['mu'])/st['sd'],-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];logits=torch.tensor(prior,dtype=torch.float32)+delta;p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:1]),logits],1),1).numpy()
  incs.append(conditioned(project(p[:,1:],support),st['minimums'][g['behavior_family'][0]]))
 return .25*g['base'].to_numpy()*support.any(1)+.25*inclusion(*ca.T)+.5*np.mean(incs,0)
def main():
 ROOT.mkdir(exist_ok=True);full,d,a,ac=data();tx,tc=design(d,a,ac);fv=d['fold'].to_numpy();sub=d['subtype'].to_numpy();amax=tx[:,tc.index('players_active_max')];q=pl.read_parquet('artifacts/evidence_session70_active_pressure/event_oof.parquet');r33=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet');parts=[];audits=[]
 for f in range(4):
  levels=[int(np.unique(amax[(fv!=f)&(sub==k)])[0]) for k in [1,2]];sup=a.group_by('pair_id','hand_id').agg(*[(C('players_active')==v).any().alias('support'+str(k)) for k,v in enumerate(levels)]);raw=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time');z=d.filter(C('fold')==f).join(raw,on=['pair_id','hand_id'],validate='1:1').join(q.select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1').join(sup,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(C('support0','support1').fill_null(False));models=[]
  for seed in [1010,2020]:
   st=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(st['state_dict']);m.eval();models.append((st,m))
  for _,g in z.group_by('pair_id'):
   g=g.sort('time','hand_id');s=g.select('support0','support1').to_numpy();p=score(g,models,s);control=score(g,models,np.ones_like(s));parts.append(g.select('pair_id','hand_id').with_columns(pl.Series('supported',p),pl.Series('control',control)));audits.append({'pair_id':g['pair_id'][0],'fold':f,'primary_supported':int(s[:,0].sum()),'secondary_supported':int(s[:,1].sum()),'listed_supported':int(((g['evidence'].to_numpy()==1)&s.any(1)).sum()),'unsupported_score_max':float(np.max(p[~s.any(1)],initial=0))})
 old=pl.read_parquet('artifacts/evidence_session70_active_pressure/background_oof.parquet').select('pair_id','hand_id',C('cat_and_joint').alias('active70'));z=old.join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1');controlerr=float(z.filter(C('control').is_not_null()).select((C('control')-C('active70')).abs().max()).item());assert controlerr<1e-12;z=z.with_columns(pl.coalesce('supported','active70').alias('supported')).join(r33.select('pair_id','hand_id',C('equal').alias('r33'),'full'),on=['pair_id','hand_id'],validate='1:1').with_columns(((C('supported')+C('full'))*.5).alias('supported_r33_recipe'));z.write_parquet(ROOT/'oof.parquet');names=['r33','active70','supported','supported_r33_recipe'];r,_=compare(ROOT/'oof.parquet',names,'supported_list');report={'method':__doc__,'control_score_replay_error':controlerr,'results':{n:{'MAP':r[n].mean(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows())} for n in names}};(ROOT/'report.json').write_text(json.dumps(report,indent=2));(ROOT/'support_audit.json').write_text(json.dumps(audits,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
