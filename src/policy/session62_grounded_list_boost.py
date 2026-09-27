\
\
\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
os.environ['BOOST_KINDS']='compact,full'
from pathlib import Path
import numpy as np,polars as pl,torch
import session11_list_boost as b
from session11_conditional_family import log_count_at_least
from session48_candidate_selection import physical
from session59_pressure_equity import COLS as PRECISION_COLS
from session8_count_conditioning import conditioned
ROOT=Path('artifacts/evidence_session62_grounded_list_boost');C=pl.col
MINIMUM=None;PCOLS=None;ORIGINAL_PACK=b.pack;ORIGINAL_DATA=b.hand_data

def grounded(d):
 x,cols=physical(d);fold=x[:,cols.index('fold_present')]>0;post=fold&(x[:,cols.index('street_no')]>0);gap=x[:,cols.index('made_category')].astype(float)+x[:,cols.index('made_kicker')]-x[:,cols.index('bet_made_category')]-x[:,cols.index('bet_made_kicker')];current=np.full((len(d),2),-2,np.float32);current[post,0]=np.sign(gap[post]);current[post,1]=gap[post]
 a=pl.read_parquet('artifacts/evidence_session57_isolation_pressure/pressure_actions.parquet');precise=np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x'];assert len(a)==len(precise);a=a.select('pair_id','hand_id').with_columns(*[pl.Series(c,precise[:,i]) for i,c in enumerate(PRECISION_COLS)]);pcols=[c+'_'+stat for stat in ['mean','max'] for c in PRECISION_COLS]+['pressure_count'];p=a.group_by('pair_id','hand_id').agg(*[getattr(C(c),stat)().alias(c+'_'+stat) for stat in ['mean','max'] for c in PRECISION_COLS],pl.len().alias('pressure_count'));p=d.select('pair_id','hand_id').join(p,on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(C('pressure_count').fill_null(0)).select(pcols).fill_null(-2).to_numpy();out=np.column_stack([x,current,p]).astype(np.float32);names=['grounded_'+c for c in cols+['pair_current_order','pair_current_gap']+pcols];assert len(set(names))==len(names) and np.isfinite(out).all();return out,names

def hand_data():
 global PCOLS
 d=ORIGINAL_DATA();x,PCOLS=grounded(d);ROOT.mkdir(exist_ok=True);np.savez_compressed(ROOT/'grounded_features.npz',x=x);(ROOT/'grounded_columns.json').write_text(json.dumps(PCOLS,indent=2));return d.with_columns(*[pl.Series(c,x[:,i]) for i,c in enumerate(PCOLS)])
def pack(d,f,ignored_cols):
 global MINIMUM
 values=ORIGINAL_PACK(d,f,PCOLS);groups,x,big,P,M,T,V,D,fv,bid,pos=values;fam=np.array([g['behavior_family'][0] for g in groups]);tr=(fv!=f)&V.any(1).numpy();mins={k:int(D.numpy()[tr&(fam==k)].min()) for k in set(fam)};MINIMUM=torch.tensor([mins[k] for k in fam]);(ROOT/f'minimums_fold{f}.json').write_text(json.dumps(mins,indent=2));return values
def objective(P,A,M,T,V,D,train):
 z=b.list_nll(P[train]+A[train],T[train],V[train],D[train])+log_count_at_least(P[train]+A[train],M[train],MINIMUM[train])/D[train];reg=(A[train].square().sum(2)*M[train]).sum(1)/M[train].sum(1);return z.sum()+b.CONFIG['ridge']*reg.sum()
def setup():
 ROOT.mkdir(exist_ok=True);b.ROOT=ROOT;b.CONFIG=dict(b.CONFIG);b.CONFIG.update({'input_root':'artifacts/evidence_session55_current_nested','objective':'family-conditional exact observed-list NLL / truth count plus mean residual square','selection':__doc__});b.hand_data=hand_data;b.pack=pack;b.objective=objective
def main():
 setup();proof=json.load(open('artifacts/evidence_session55_current_nested/input_verification.json'));assert proof['saved_models_replayed']==175 and proof['event_probability_error']==0 and proof['all_training_prediction_and_outer_pool_sets_disjoint'];b.main();d=hand_data();parts=[]
 for f in range(4):
  groups,x,big,P,M,T,V,D,fv,bid,pos=pack(d,f,None);mins=json.load(open(ROOT/f'minimums_fold{f}.json'));pred={}
  for kind in ['compact','full']:
   X=x if kind=='compact' else np.column_stack([x,big]);X=np.nan_to_num(X,nan=0,posinf=1e6,neginf=-1e6);model=b.joblib.load(ROOT/f'list_boost_{kind}_fold{f}.joblib');model['minimums']=mins;model['grounded_columns']=PCOLS;model['columns']=[] if kind=='compact' else PCOLS;model['feature_count']=X.shape[1];b.joblib.dump(model,ROOT/f'list_boost_{kind}_fold{f}.joblib',compress=3);z=P.numpy().copy();z[bid,pos]+=b.infer(model,X);pred[kind]=z
  for i in np.flatnonzero(fv==f):
   g=groups[i];q=g.select('pair_id','hand_id','fold','evidence','r29')
   for kind,z in pred.items():
    p=torch.softmax(torch.tensor(np.column_stack([np.zeros(len(g),np.float32),z[i,:len(g)]])),1).numpy();inc=conditioned(p[:,1:],mins[g['behavior_family'][0]]);score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;q=q.with_columns(pl.Series(kind,score),pl.Series(kind+'_inclusion',inc))
   parts.append(q)
 pl.concat(parts).write_parquet(ROOT/'conditional_boost_oof.parquet')
if __name__=='__main__':main()
