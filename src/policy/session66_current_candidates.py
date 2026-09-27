\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier,Pool
from scipy.special import logit,expit
from session8_data import hand_data
from session48_candidate_selection import design,CONFIG
ROOT=Path('artifacts/evidence_session66_current_candidates');INPUT='artifacts/evidence_session55_current_nested';C=pl.col
def main():
 ROOT.mkdir(exist_ok=True);proof=json.load(open(Path(INPUT)/'input_verification.json'));assert proof['event_probability_error']==0 and proof['all_training_prediction_and_outer_pool_sets_disjoint'];(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'input_root':INPUT,**CONFIG,'physical_source':'verified62 grounded_features.npz;134raw fields','transport':'fixed learned logit residual onR33 within the analytic shortlist, prior-shift experiment'},indent=2));d=hand_data();px=np.load('artifacts/evidence_session62_grounded_list_boost/grounded_features.npz')['x'];assert len(px)==len(d);fv=d['fold'].to_numpy();y=d['evidence'].to_numpy();old=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id','equal'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['equal'].to_numpy();pred={k:np.zeros(len(d)) for k in ['analytic','compact','grounded','compact_transport','grounded_transport']};audit=[];start=time.time()
 for f in range(4):
  x,prior,selected,minimums=design(d,f,INPUT);tr=(fv!=f)&selected;va=fv==f;sv=va&selected;p0=logit(prior.clip(1e-7,1-1e-7));pred['analytic'][va]=prior[va];d.select('pair_id','hand_id').with_columns(pl.Series('prior',prior),pl.Series('selected',selected)).write_parquet(ROOT/f'design_fold{f}.parquet');assert not(tr&va).any()
  for kind in ['compact','grounded']:
   xx=x if kind=='compact' else np.column_stack([x,px]);m=CatBoostClassifier(iterations=400,depth=4,learning_rate=.03,l2_leaf_reg=20,random_seed=4810+f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(Pool(xx[tr],y[tr],baseline=p0[tr]));m.save_model(str(ROOT/f'{kind}_fold{f}.cbm'));delta=m.predict(xx[sv],prediction_type='RawFormulaVal',thread_count=2);pred[kind][va]=prior[va];pred[kind][sv]=expit(p0[sv]+delta);pred[kind+'_transport'][va]=old[va];pred[kind+'_transport'][sv]=expit(logit(old[sv].clip(1e-7,1-1e-7))+delta);audit.append({'fold':f,'kind':kind,'features':xx.shape[1],'training_candidates':int(tr.sum()),'positive_training_candidates':int(y[tr].sum()),'heldout_candidates':int(sv.sum()),'descriptive_heldout_positive_candidates':int(y[sv].sum()),'validation_overlap':0,'minimums':minimums});print('current shortlist',f,kind,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id').with_columns(*[pl.Series(k,v) for k,v in pred.items()]).write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
