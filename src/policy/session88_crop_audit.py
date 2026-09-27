\
\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from session8_data import hand_data
from session11_build_candidate import models as correction_models
from session11_conditional_family import features
from session6_priority import inclusion
from session8_count_conditioning import conditioned
from session65_list_pressure_inference import tree_delta,GROUND_COLUMNS
ROOT=Path('artifacts/evidence_session88_crop_audit');C=pl.col
WINDOWS={'full':(0.,.6),'first_2000':(0.,.4),'last_2000':(.2,.6)}
def data():
 d=hand_data();x=np.load('artifacts/evidence_session62_grounded_list_boost/grounded_features.npz')['x'];assert len(x)==len(d);d=d.with_columns(*[pl.Series(n,x[:,i]) for i,n in enumerate(GROUND_COLUMNS)])
 for folder,prefix in [('evidence_session50_matchup/current','r32'),('evidence_session59_pressure_equity','pressure')]:
  z=pl.read_parquet('artifacts/'+folder+'/event_oof.parquet').select('pair_id','hand_id',C('bg_primary').alias(prefix+'_primary'),C('bg_secondary').alias(prefix+'_secondary'));d=d.join(z,on=['pair_id','hand_id'],validate='1:1')
 old=[]
 for f in range(4):old.append(pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time'))
 d=d.join(pl.concat(old),on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33_full')),on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id','conditional_family'),on=['pair_id','hand_id'],validate='1:1').join(pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score'),on='pair_id',validate='m:1');return d
def refresh(g,prefix=None):
 if prefix:g=g.with_columns(C(prefix+'_primary').alias('cat_primary'),C(prefix+'_secondary').alias('cat_secondary'))
 ca=g.select('cat_primary','cat_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];ci=inclusion(*ca.T);ji=inclusion(*((ca+hp)/2).T);return g.with_columns(pl.Series('cat_inclusion',ci),pl.Series('joint_inclusion',ji),pl.Series('r29',.25*g['base'].to_numpy()+.25*ci+.5*ji))
def score(g,f,nn,tree,refresh_legacy=False):
 old=refresh(g) if refresh_legacy else g;x,_=features(old);ca=g.select('pressure_primary','pressure_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];jp=(ca+hp)/2;prior=np.log(np.maximum(jp,1e-6))-np.log(np.maximum(1-jp.sum(1),1e-6))[:,None];incs=[]
 for m,mu,sd,mins in nn[f]:
  with torch.no_grad():delta=m(torch.tensor(np.clip((x-mu)/sd,-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];z=torch.tensor(prior,dtype=torch.float32)+delta;p=torch.softmax(torch.cat([torch.zeros_like(z[:,:1]),z],1),1).numpy()
  incs.append(conditioned(p[:,1:],mins[g['behavior_family'][0]]))
 ps=.25*g['base'].to_numpy()+.25*inclusion(*ca.T)+.5*np.mean(incs,0);q=refresh(g,'r32');x,prior=features(q);xx=np.nan_to_num(np.column_stack([x,q.select(GROUND_COLUMNS).to_numpy()]),nan=0,posinf=1e6,neginf=-1e6);z=prior+tree_delta(tree[f],xx);p=torch.softmax(torch.tensor(np.column_stack([np.zeros(len(g),np.float32),z])),1).numpy();inc=conditioned(p[:,1:],tree[f]['minimums'][g['behavior_family'][0]]);ts=.25*q['base'].to_numpy()+.25*q['cat_inclusion'].to_numpy()+.5*inc;return (ps+ts)/2
def ap(g,col):
 y=g.sort(col,'hand_id',descending=[True,False])['evidence'].to_numpy()[:5];return float((y*y.cumsum()/np.arange(1,len(y)+1)).sum()/min(5,g['evidence'].sum()))
def main():
 ROOT.mkdir(exist_ok=True);d=data();nn=correction_models('conditional_family');tree=[joblib.load(f'artifacts/evidence_session62_grounded_list_boost/list_boost_full_fold{f}.joblib') for f in range(4)];rows=[];coverage=[];replay=0
 for (pid,),g in d.group_by('pair_id'):
  g=g.sort('time','hand_id');f=g['fold'][0];full=score(g,f,nn,tree);replay=max(replay,float(abs(full-g['r33_full'].to_numpy()).max()));truth=g['evidence'].sum();time_index=np.rint(g['time'].to_numpy()*5000).astype(int)
  for window,(lo,hi) in WINDOWS.items():
   mask=(time_index>=round(lo*5000))&(time_index<round(hi*5000));z=g.filter(pl.Series(mask));retained=int(z['evidence'].sum());coverage.append({'pair_id':pid,'window':window,'family':g['behavior_family'][0],'fold':f,'hands':len(z),'truth_total':int(truth),'truth_retained':retained,'complete_list_retained':retained==truth})
   if retained!=truth or not len(z):continue
                                                                          
   coord=z if window=='full' else z.with_columns(((C('time')-lo)/(hi-lo)).alias('relative_time'));fresh=score(coord,f,nn,tree,refresh_legacy=window!='full');frozen=z['r33_full'].to_numpy()
   if g['risk_score'][0]<.05:fresh=frozen=z['conditional_family'].to_numpy()
   z=z.with_columns(pl.Series('frozen',frozen),pl.Series('recomputed',fresh));rows.append({'pair_id':pid,'table_id':g['table_id'][0],'family':g['behavior_family'][0],'fold':f,'window':window,'hands':len(z),'frozen':ap(z,'frozen'),'recomputed':ap(z,'recomputed')})
 assert replay<1e-7;result=pl.DataFrame(rows);cov=pl.DataFrame(coverage);result.write_parquet(ROOT/'pair_comparison.parquet');cov.write_parquet(ROOT/'coverage.parquet');reports={}
 for (window,),q in result.group_by('window'):
  pool=q.group_by('table_id').agg(C('frozen','recomputed').sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(8812).integers(0,len(pool),(5000,len(pool)));delta=pool['recomputed'].to_numpy()-pool['frozen'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);reports[window]={'pairs':len(q),'frozen_MAP':q['frozen'].mean(),'recomputed_MAP':q['recomputed'].mean(),'gain':q['recomputed'].mean()-q['frozen'].mean(),'CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':q.group_by('fold').agg(C('frozen','recomputed').mean()).sort('fold').to_dicts(),'families':q.group_by('family').agg(pl.len(),C('frozen','recomputed').mean()).to_dicts()}
 report={'method':__doc__,'full_R33_score_replay_error':replay,'coverage':cov.group_by('window').agg(pl.len(),C('complete_list_retained').sum(),(C('truth_retained')==0).sum().alias('zero_retained')).to_dicts(),'results':reports,'limits':['Complete-list crop selection conditions on truth and changes the evaluated population.','Per-hand features and donor posteriors are frozen; this isolates list-stage shortening.','Cropped results are not comparable to full MAP across different pair sets.','No new model, candidate, or leaderboard result.']};(ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
