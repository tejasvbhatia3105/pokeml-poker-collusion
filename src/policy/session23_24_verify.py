import os,json,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import polars as pl,numpy as np
from threadpoolctl import threadpool_limits
from session6_priority_model import load_models as cats
from session7_model import load_models as events
from session8_data import hand_data,targets
from session24_pseudo_rate_control import group_weights
C=pl.col
ROOT=Path('artifacts/evidence_session23_more_pseudo')
def main():
 d=hand_data();cm,cols=cats('priority_ordered');hm,_=events('hist_eventblend');report=[];weights=[]
 reg=pl.read_csv('data/evaluation_pairs.csv');folds=json.load(open('artifacts/policy/table_folds.json'))
 with threadpool_limits(limits=2):
  for f in range(4):
   odd=pl.read_parquet(ROOT/f'odd_outer{f}.parquet');gates=pl.read_parquet(ROOT/f'odd_gates_fold{f}.parquet')
   even=pl.read_parquet(list(Path(f'artifacts/evidence_session12/unlabelled_pseudo/outer{f}').glob('T*.parquet'))).join(pl.read_parquet(f'artifacts/evidence_session12/pseudo_pair_gate/gate_fold{f}.parquet').filter(C('keep')).select('pair_id'),on='pair_id',validate='m:1')
   assert set(odd['pair_id'])<=set(reg['pair_id']) and not set(odd['pair_id'])&set(even['pair_id'])
   assert all(hashlib.sha256(p.encode()).digest()[0]%2==1 for p in odd['pair_id'].unique())
   assert (odd['teacher_fold']==f).all() and (odd['pool_fold']!=f).all()
   assert gates.select((C('table_id').replace_strict(folds)==C('pool_fold')).all()).item()
   assert (gates['pair_risk']>=.9).all();assert not len(odd.join(d.select('pair_id','hand_id'),on=['pair_id','hand_id']))
   q=pl.concat([even,odd],how='vertical_relaxed');assert q.select('pair_id','hand_id').n_unique()==len(q)
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    z=odd.filter(C('behavior_family')==fam);ids=sorted(z['pair_id'].unique())[:2];z=z.filter(C('pair_id').is_in(ids));x=z.select(cols).to_numpy();err=0.
    for k,name in enumerate(['primary','secondary']):
     cp=cm[fam][f][k].predict_proba(x,thread_count=2)[:,1];hp=hm[fam][f][1][k].predict_proba(x)[:,1];p=.5*(cp+hp);err=max(err,float(abs(p-z['pseudo_'+name].to_numpy()).max()));e=((cp>.9)&(hp>.9))|((cp<.02)&(hp<.02));assert np.array_equal(e,z['eligible_'+name].to_numpy())
    assert err<1e-5
    report.append({'fold':f,'family':fam,'sampled_pairs':len(ids),'sampled_rows':len(z),'teacher_probability_error':err})
    a,b,ea,eb,va=targets(d,f,fam);z=q.filter(C('behavior_family')==fam)
    for k,(y,e) in enumerate([(a,ea),(b,eb)]):
     name=['primary','secondary'][k];p=z.filter(C('eligible_'+name))['pseudo_'+name].to_numpy();w,info=group_weights(p,int(e.sum()),float(y[e].mean()));assert not (e&va).any();assert np.all(w>=0);assert abs(w.sum()-.5*e.sum())<1e-8;weights.append({'fold':f,'family':fam,'head':k+1,**info})
 saved=json.load(open('artifacts/evidence_session24_rate_control/audit.json'))
 for a,b in zip(weights,saved):
  for key in ['fold','family','head','high','low','fallback']:assert a[key]==b[key]
  for key in ['weighted_pseudo_rate','original_target_rate','high_weight_fraction']:
   if key in a:assert abs(a[key]-b[key])<1e-12
 out={'sampled_teacher_replays':report,'weight_replays':len(weights),'isolation':'registry membership, odd/even disjointness, teacher fold, actual pool fold, no original hand overlap all pass','scope':'sampled original event teacher replay; pair gate model ancestry previously verified in session12; not an independent rebuild of every raw feature'}
 (ROOT/'pseudo_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
