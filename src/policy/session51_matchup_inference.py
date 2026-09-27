import os,json
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
import session46_paired_inference as paired
from session46_paired_inference import score_pair,old_score,NAMES
from session50_matchup import ROOT as FEATURE_ROOT,compute
KIND=os.environ.get('MATCHUP_KIND','current');assert KIND in ['full','outcomes','current']
MODEL_ROOT=FEATURE_ROOT if KIND=='full' else FEATURE_ROOT/KIND
ROOT=Path('artifacts/evidence_session51_inference')/KIND
SELECTED=list(range(7)) if KIND=='full' else json.load(open(MODEL_ROOT/'config.json'))['selected_columns']
def load_models():
 out=paired.load_models();paths={'direct_primary':('directed_transfer',1),'soft_primary':('soft_play',1),'iso_primary':('coordinated_isolation',1),'iso_secondary':('coordinated_isolation',2)}
 for f in range(4):
  for name,(fam,k) in paths.items():m=CatBoostClassifier();m.load_model(str(MODEL_ROOT/fam/f'event{k}_fold{f}.cbm'));out['fold'][f][name]=m
 return out
def current_from_design(a,x,cfg):
 ac=cfg['fold_columns'];ec=cfg['paired_columns'];offset=len(ac)+len(cfg['hand_columns']);gap=x[:,ac.index('made_category')]+x[:,ac.index('made_kicker')]-x[:,offset+ec.index('bet_made_category')]-x[:,offset+ec.index('bet_made_kicker')];post=a['street_no'].to_numpy()>0
 return np.column_stack([np.where(post,np.sign(gap),-2),np.where(post,gap,-2)]).astype(np.float32)
def augment(d,a,players,x,ix):
 if KIND=='current':extra=current_from_design(a,x,json.load(open('artifacts/evidence_session37_bet_fold/config.json')))
 elif len(a):_,extra=compute(d,a,players);extra=extra[:,SELECTED]
 else:extra=np.empty((0,len(SELECTED)))
 hx=np.full((len(d),len(SELECTED)),-2.);hx[a['row'].to_numpy()]=extra
 return np.column_stack([x,extra]),np.column_stack([ix,hx])
def events(d,players,models,folds=range(4)):
 return paired.events(d,players,models,folds,augment=augment)
