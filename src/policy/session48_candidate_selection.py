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
from catboost import CatBoostClassifier,Pool
from scipy.special import expit,logit
from session8_data import hand_data
from session11_conditional_family import features
from session8_count_conditioning import conditioned
C=pl.col;ROOT=Path('artifacts/evidence_session48_candidate_selection')
CONFIG={'candidate_count':12,'iterations':400,'depth':4,'learning_rate':.03,'l2_leaf_reg':20,'seed':'4810+fold','loss':'binary exact listed-hand membership among proposed candidates, prior log-odds offset','selection':'same fixed schedule; compact vs paired; no heldout stopping or candidate promotion','transport':'same learned residual applied to stronger R31 baseline, explicitly a prior-shift experiment'}

def physical(d):
 pieces=[];ac=None;ec=None
 for folder,actionpath in [('artifacts/evidence_session37_bet_fold','artifacts/evidence_session26_exact_fold/fold_actions.parquet'),('artifacts/evidence_session38_soft_bet_fold','artifacts/evidence_session38_soft_bet_fold/fold_actions.parquet'),('artifacts/evidence_session41_isolation_bet_fold','artifacts/evidence_session41_isolation_bet_fold/fold_actions.parquet')]:
  cfg=json.load(open(Path(folder)/'config.json'));nowac=cfg['fold_columns'];nowec=cfg['paired_columns'];assert ac is None or (nowac==ac and nowec==ec);ac=nowac;ec=nowec;a=pl.read_parquet(actionpath)
  if 'action_row' not in a.columns:a=a.with_row_index('action_row')
  ex=pl.read_parquet(Path(folder)/'action_features.parquet');a=a.select('pair_id','hand_id','action_row',*ac).join(ex,on='action_row',validate='1:1').drop('action_row').with_columns(pl.lit(1.).alias('fold_present'));pieces.append(a)
 p=pl.concat(pieces);assert p.select('pair_id','hand_id').n_unique()==len(p);cols=ac+ec+['fold_present'];q=d.select('pair_id','hand_id').join(p,on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(C('fold_present').fill_null(0));q=q.with_columns(C(ac+ec).fill_null(-2));return q.select(cols).to_numpy(),cols

def design(d,f,input_root='artifacts/evidence_session10/nested6'):
 q=d.join(pl.read_parquet(Path(input_root)/f'nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');counts=d.group_by('pair_id').agg(C('fold').first(),C('behavior_family').first(),C('evidence').sum().alias('n'));minimums=dict(counts.filter(C('fold')!=f).group_by('behavior_family').agg(C('n').min()).iter_rows());X=np.zeros((len(d),35),np.float32);prior=np.zeros(len(d));selected=np.zeros(len(d),bool)
 for _,g in q.group_by('pair_id'):
  g=g.sort('time','hand_id');ix=g['row'].to_numpy();x,_=features(g);ca=g.select('cat_primary','cat_secondary').to_numpy();hp=g.select('hist_primary','hist_secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=hp/np.maximum(1,hp.sum(1))[:,None];pr=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*conditioned((ca+hp)/2,minimums[g['behavior_family'][0]]);X[ix]=x;prior[ix]=pr;top=np.lexsort((g['hand_id'].to_numpy(),-pr))[:CONFIG['candidate_count']];selected[ix[top]]=True
 return X,prior,selected,minimums

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,**CONFIG},indent=2));d=hand_data();PX,pc=physical(d);np.savez_compressed(ROOT/'physical_features.npz',x=PX);(ROOT/'physical_columns.json').write_text(json.dumps(pc));fv=d['fold'].to_numpy();y=d['evidence'].to_numpy();old=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/background_oof.parquet').select('pair_id','hand_id','cat_and_joint'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['cat_and_joint'].to_numpy();pred={k:np.zeros(len(d)) for k in ['analytic','compact','paired','compact_transport','paired_transport']};audit=[];start=time.time()
 for f in range(4):
  X,pr,selected,minimums=design(d,f);tr=(fv!=f)&selected;va=fv==f;sv=va&selected;p0=logit(pr.clip(1e-7,1-1e-7));pred['analytic'][va]=pr[va];d.select('pair_id','hand_id').with_columns(pl.Series('prior',pr),pl.Series('selected',selected)).write_parquet(ROOT/f'design_fold{f}.parquet');assert not (tr&va).any()
  for kind in ['compact','paired']:
   xx=X if kind=='compact' else np.column_stack([X,PX]);m=CatBoostClassifier(iterations=400,depth=4,learning_rate=.03,l2_leaf_reg=20,thread_count=2,random_seed=4810+f,verbose=False,allow_writing_files=False);m.fit(Pool(xx[tr],y[tr],baseline=p0[tr]));m.save_model(str(ROOT/f'{kind}_fold{f}.cbm'));delta=m.predict(xx[sv],prediction_type='RawFormulaVal',thread_count=2);pred[kind][va]=pr[va];pred[kind][sv]=expit(p0[sv]+delta);pred[kind+'_transport'][va]=old[va];pred[kind+'_transport'][sv]=expit(logit(old[sv].clip(1e-7,1-1e-7))+delta);audit.append({'fold':f,'kind':kind,'training_candidates':int(tr.sum()),'training_positive_candidates':int(y[tr].sum()),'heldout_candidates':int(sv.sum()),'heldout_positive_candidates_descriptive':int(y[sv].sum()),'validation_overlap':0,'minimums':minimums});print('candidate selection',f,kind,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id').with_columns(*[pl.Series(k,v) for k,v in pred.items()]).write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
