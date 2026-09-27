\
\
\
\
\
\
\
\
import os,json,itertools,ctypes,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from cards import CARD
from session8_data import hand_data,targets
from session35_fold_likelihood import labels,OLD
from session29_shared_fold_witness import target
from session42_precision import family_data
from session41_isolation_bet_fold import designs
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session50_matchup');C=pl.col;PERMS=list(itertools.permutations(range(4)));COLS=['pair_equity','pair_win','pair_tie','pair_current_comparison','pair_own_improvement','pair_partner_improvement','pair_current_rank_gap']
def canonical(cs):
 return min(tuple(sorted(4*(c//4)+s[c%4] for c in cs[:2]))+tuple(sorted(4*(c//4)+s[c%4] for c in cs[2:4]))+tuple(sorted(4*(c//4)+s[c%4] for c in cs[4:] if c>=0))+tuple(-1 for c in cs[4:] if c<0) for s in PERMS)
def compute(d,a,players=None):
 labs=(pl.read_csv('data/development_labels.csv') if players is None else players).select('pair_id','player_1','player_2');z=a.join(labs,on='pair_id',validate='m:1',maintain_order='left');states=np.full((len(a),9),-1,np.int8)
 for (table,),q in d.group_by('table_id'):
  local=z.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');needed=local.select('hand_id').unique();s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(needed,on='hand_id',how='semi');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(needed,on='hand_id',how='semi');holes={(hid,p):(CARD[c1],CARD[c2]) for hid,p,c1,c2 in s.select('hand_id','player_id','hole_card_1','hole_card_2').iter_rows()};boards={hid:[CARD[c] for c in b.split()] for hid,b in h.select('hand_id','board_cards').iter_rows()}
  for r in local.to_dicts():
   actor=r['player_1'] if r['actor']==0 else r['player_2'];partner=r['player_2'] if r['actor']==0 else r['player_1'];st=int(r['street_no']);nb=0 if st==0 else st+2;cs=list(holes[r['hand_id'],actor])+list(holes[r['hand_id'],partner])+boards[r['hand_id']][:nb]+[-1]*(5-nb);states[r['action_row']]=canonical(cs)
 assert np.all(states[:,:4]>=0);return states,evaluate(states)
def evaluate(states):
 states=np.ascontiguousarray(states,np.int8);lib=ctypes.CDLL(str((ROOT/'matchup.dylib').resolve()));lib.poker_pair_matchup.argtypes=[ctypes.POINTER(ctypes.c_int8),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)];out=np.empty((len(states),7),np.float32);lib.poker_pair_matchup(states.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),len(states),2048,out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)));assert np.isfinite(out).all() and np.all((out[:,:3]>=0)&(out[:,:3]<=1));return out
def load(family):
 if family!='coordinated_isolation':
  d,a,x,cols,*_=family_data(family);return d,a,x,cols
 d=hand_data().filter(C('behavior_family')==family).drop('row').with_row_index('row');folder=Path('artifacts/evidence_session41_isolation_bet_fold');a=pl.read_parquet(folder/'fold_actions.parquet');cfg=json.load(open(folder/'config.json'));ex=pl.read_parquet(folder/'action_features.parquet');xs,hc,ec=designs(d,a,cfg['fold_columns'],ex);return d,a,xs['paired_fold'],hc+cfg['fold_columns']+['fold_present']+ec
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'columns':COLS,'event_model':'Cat400 depth5 lr.035 L2 8 seeds6311+11fold+head','baseline':'session41 paired_fold Cat-and-joint'},indent=2));full=hand_data();base=pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/event_oof.parquet');parts=[];audit=[];start=time.time()
 for family in ['directed_transfer','soft_play','coordinated_isolation']:
  root=ROOT/family;root.mkdir(exist_ok=True);d,a,x,cols=load(family);states,ex=compute(d,a);np.savez_compressed(root/'features.npz',states=states,x=ex);g=a['row'].to_numpy();r=a['actor'].to_numpy();iso=family=='coordinated_isolation';xx=np.full((len(d),7),-2.) if iso else ex
  if iso:xx[g]=ex
  x=np.column_stack([x,xx]);pp=np.zeros((len(d),2));fv=d['fold'].to_numpy();fam=full['behavior_family'].to_numpy()==family
  for f in range(4):
   if iso:
    p1,p2,e1,e2,_=targets(full,f,family);ys=np.column_stack([p1,p2])[fam];es=np.column_stack([e1,e2])[fam];va=fv==f
   elif family=='directed_transfer':y,tr,va=labels(d,a,f)
   else:y0,e,_=target(d,f);y=y0[g];tr=e[g];va=fv[g]==f
   for k in range(2 if iso else 1):
    if iso:y=ys[:,k];tr=es[:,k]
    assert not(tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f+k,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(root/f'event{k+1}_fold{f}.cbm'));p=m.predict_proba(x[va],thread_count=2)[:,1]
    if iso:pp[va,k]=p
    else:pp[g[va],r[va]]=p
    audit.append({'family':family,'fold':f,'head':k+1,'training_count':int(tr.sum()),'positive_count':int(y[tr].sum()),'validation_overlap':0})
   print('matchup',family,f,round(time.time()-start,1),flush=True)
  if iso:new=pp
  else:
   dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if family=='directed_transfer' else np.ones_like(pp);new=(pp*dw).sum(1)[:,None]
  parts.append(d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',new[:,0]),pl.Series('new_secondary',new[:,1] if iso else np.full(len(d),np.nan))))
 q=base.join(pl.concat(parts),on=['pair_id','hand_id'],validate='1:1').with_columns(C('new_primary').alias('bg_primary'),pl.when(C('new_secondary').is_nan()).then(C('bg_secondary')).otherwise(C('new_secondary')).alias('bg_secondary')).drop('new_primary','new_secondary');q.write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':main()
