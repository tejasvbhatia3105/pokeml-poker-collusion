\
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
from catboost import CatBoostRanker,Pool
from scipy.special import expit,logit
from session48_candidate_selection import physical,design
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session49_candidate_ranking')

def training_pool(d,xx,y,pr,tr):
 ix=np.flatnonzero(tr); gids=d['pair_id'].to_numpy()[ix]; weights=np.ones(len(ix))
 for pid in np.unique(gids):
  j=np.flatnonzero(gids==pid); k=y[ix[j]].sum(); weights[j]=1/max(1,k*(len(j)-k))
                                                                    
 return Pool(xx[ix],y[ix],group_id=gids,group_weight=weights,baseline=logit(pr[ix].clip(1e-7,1-1e-7)))

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'iterations':400,'depth':4,'learning_rate':.03,'l2_leaf_reg':20,'loss':'PairLogit','seed':'4810+fold','candidate_count':12},indent=2));d=hand_data();PX,pc=physical(d);fv=d['fold'].to_numpy();y=d['evidence'].to_numpy();old=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/background_oof.parquet').select('pair_id','hand_id','cat_and_joint'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['cat_and_joint'].to_numpy();pred={k:np.zeros(len(d)) for k in ['compact','paired','compact_transport','paired_transport']};audit=[];start=time.time()
 for f in range(4):
  X,pr,selected,minimums=design(d,f);tr=(fv!=f)&selected;va=fv==f;sv=va&selected;assert not (tr&va).any()
  for kind in ['compact','paired']:
   xx=X if kind=='compact' else np.column_stack([X,PX]);pool=training_pool(d,xx,y,pr,tr);m=CatBoostRanker(iterations=400,depth=4,learning_rate=.03,l2_leaf_reg=20,loss_function='PairLogit',thread_count=2,random_seed=4810+f,verbose=False,allow_writing_files=False);m.fit(pool);m.save_model(str(ROOT/f'{kind}_fold{f}.cbm'));delta=m.predict(xx[sv],thread_count=2);pred[kind][va]=pr[va];pred[kind][sv]=expit(logit(pr[sv].clip(1e-7,1-1e-7))+delta);pred[kind+'_transport'][va]=old[va];pred[kind+'_transport'][sv]=expit(logit(old[sv].clip(1e-7,1-1e-7))+delta);audit.append({'fold':f,'kind':kind,'training_candidates':int(tr.sum()),'training_positive_candidates':int(y[tr].sum()),'heldout_candidates':int(sv.sum()),'validation_overlap':0,'minimums':minimums});print('candidate ranking',f,kind,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id').with_columns(*[pl.Series(k,v) for k,v in pred.items()]).write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
