\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from catboost import CatBoostClassifier
from session5_multiway_features import build
from session37_bet_fold import paired_features
from session25_persistent_actor import actor_features,pair_features
from session11_build_candidate import models as correction_models,NAMES,score as old_score
from session11_conditional_family import features
from session6_priority import inclusion
from session8_count_conditioning import conditioned
C=pl.col;ROOT=Path('artifacts/evidence_session46_inference')

def load_models():
 out={'correction':correction_models('conditional_family'),'fold':[]};paths={'actor':'artifacts/evidence_session25_persistent_actor/actor_fold{f}.cbm','direct_primary':'artifacts/evidence_session37_bet_fold/paired_hand/primary_fold{f}.cbm','direct_secondary':'artifacts/evidence_session25_persistent_actor/oriented/event2_directed_transfer_fold{f}.cbm','soft_primary':'artifacts/evidence_session38_soft_bet_fold/paired_soft/primary_fold{f}.cbm','iso_primary':'artifacts/evidence_session41_isolation_bet_fold/paired_fold/event1_fold{f}.cbm','iso_secondary':'artifacts/evidence_session41_isolation_bet_fold/paired_fold/event2_fold{f}.cbm'}
 for f in range(4):
  part={}
  for k,path in paths.items():m=CatBoostClassifier();m.load_model(path.format(f=f));part[k]=m
  out['fold'].append(part)
 out['cfg']=json.load(open('artifacts/evidence_session37_bet_fold/config.json'));return out

def action_design(d,players,cfg):
 assert d['table_id'].n_unique()==1;table=d['table_id'][0];query=d.select('pair_id','hand_id').join(players.select('pair_id','player_1','player_2'),on='pair_id',validate='m:1');raw=build(table,query,return_actions=True,only_partner_folds=True);ac=cfg['fold_columns'];ec=cfg['paired_columns'];a=raw.select('pair_id','hand_id','player_id',*ac).join(players.select('pair_id','player_2'),on='pair_id',validate='m:1').with_columns((C('player_id')==C('player_2')).cast(pl.Int8).alias('actor')).drop('player_id','player_2').join(d.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],validate='m:1').sort('pair_id','hand_id','action_no').with_row_index('action_row');assert a.select('pair_id','hand_id').n_unique()==len(a)
 if len(a):ex,_=paired_features(d,a,players=players);extra=ex.select(ec).to_numpy()
 else:extra=np.empty((0,len(ec)))
 hx=d.select(cfg['hand_columns']).to_numpy();ax=a.select(ac).to_numpy();g=a['row'].to_numpy();x=np.column_stack([ax,hx[g],extra]);rawhand=np.full((len(d),len(ac)+1),-2.);rawhand[:,-1]=0;rawhand[g,:-1]=ax;rawhand[g,-1]=1;pairedhand=np.full((len(d),len(ec)),-2.);pairedhand[g]=extra;iso=np.column_stack([hx,rawhand,pairedhand]);return a,x,iso,hx

def events(d,players,models,folds=range(4),augment=None):
 d=d.sort('pair_id','time','hand_id')
 if 'row' in d.columns:d=d.drop('row')
 d=d.with_row_index('row');cfg=models['cfg'];a,x,ix,hx=action_design(d,players,cfg)
 if augment is not None:x,ix=augment(d,a,players,x,ix)
 g=a['row'].to_numpy();actor=a['actor'].to_numpy();fam=d['behavior_family'].to_numpy();direct=np.flatnonzero(fam=='directed_transfer');soft_action=fam[g]=='soft_play';dir_action=fam[g]=='directed_transfer';iso=fam=='coordinated_isolation';dd=d[direct].with_row_index('local_row') if len(direct) else None
 if len(direct):
  AX=actor_features(dd,players);groups=[z['local_row'].to_numpy() for _,z in dd.group_by('pair_id',maintain_order=True)];PX=pair_features(AX,groups)
 for f in folds:
  m=models['fold'][f];primary=np.zeros(len(d));secondary=d[f'cat_secondary_{f}'].to_numpy().copy()
  if len(direct):
   pr=m['actor'].predict_proba(PX.reshape(-1,PX.shape[-1]),thread_count=2)[:,1].reshape(-1,2);pr/=pr.sum(1)[:,None];dw=np.zeros((len(d),2))
   for group,p in zip(groups,pr):dw[direct[group]]=p
   pp=np.zeros((len(d),2))
   if dir_action.any():pp[g[dir_action],actor[dir_action]]=m['direct_primary'].predict_proba(x[dir_action],thread_count=2)[:,1]
   primary[direct]=(pp*dw).sum(1)[direct];sec=np.column_stack([m['direct_secondary'].predict_proba(np.column_stack([hx[direct],AX[:,r]]),thread_count=2)[:,1] for r in range(2)]);secondary[direct]=(sec*dw[direct]).sum(1)
  if soft_action.any():primary[g[soft_action]]=m['soft_primary'].predict_proba(x[soft_action],thread_count=2)[:,1]
  if iso.any():primary[iso]=m['iso_primary'].predict_proba(ix[iso],thread_count=2)[:,1];secondary[iso]=m['iso_secondary'].predict_proba(ix[iso],thread_count=2)[:,1]
  d=d.with_columns(pl.Series(f'new_primary_{f}',primary),pl.Series(f'new_secondary_{f}',secondary))
 return d

def score_pair(g,f,models):
 q=g.with_columns(*[C(f'{n}_{f}').alias(n) for n in NAMES]);ca=q.select(f'new_primary_{f}',f'new_secondary_{f}').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=q.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];jp=(ca+hp)/2;prior=np.log(np.maximum(jp,1e-6))-np.log(np.maximum(1-jp.sum(1),1e-6))[:,None];x,_=features(q);incs=[]
 for m,mu,sd,minimums in models['correction'][f]:
  with torch.no_grad():delta=m(torch.tensor(np.clip((x-mu)/sd,-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];z=torch.tensor(prior,dtype=torch.float32)+delta;p=torch.softmax(torch.cat([torch.zeros_like(z[:,:1]),z],1),1).numpy()
  incs.append(conditioned(p[:,1:],minimums[g['behavior_family'][0]]))
 return .25*q['base'].to_numpy()+.25*inclusion(*ca.T)+.5*np.mean(incs,0)
