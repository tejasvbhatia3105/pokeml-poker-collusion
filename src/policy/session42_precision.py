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
from cards import CARD,preflop_table,rank
from session8_data import hand_data
from session35_fold_likelihood import labels,OLD,EXACT
from session29_shared_fold_witness import target
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session42_precision');BASE=Path('artifacts/evidence_session38_soft_bet_fold/paired_soft');PERMS=list(itertools.permutations(range(4)))
def canonical(cards):
 return min(tuple(sorted(4*(c//4)+s[c%4] for c in cards[:2]))+tuple(sorted(4*(c//4)+s[c%4] for c in cards[2:] if c>=0))+tuple(-1 for c in cards[2:] if c<0) for s in PERMS)
def family_data(family):
 d=hand_data().filter(C('behavior_family')==family).drop('row').with_row_index('row')
 if family=='directed_transfer':
  a=pl.read_parquet(EXACT/'fold_actions.parquet').with_row_index('action_row');folder=Path('artifacts/evidence_session37_bet_fold');cfg=json.load(open(folder/'config.json'));ac=cfg['fold_columns'];hc=cfg['hand_columns'];ec=cfg['paired_columns']
 else:
  folder=Path('artifacts/evidence_session38_soft_bet_fold');a=pl.read_parquet(folder/'fold_actions.parquet');cfg=json.load(open(folder/'config.json'));ac=cfg['fold_columns'];hc=cfg['hand_columns'];ec=cfg['paired_columns']
 ex=pl.read_parquet(folder/'action_features.parquet').sort('action_row');g=a['row'].to_numpy();x=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],ex.select(ec).to_numpy()]);cols=ac+hc+ec;return d,a,x,cols,ac,hc,ec
def compute(d,a):
 labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');z=a.join(labs,on='pair_id',validate='m:1',maintain_order='left');states=[];where=[];PF=preflop_table();values=np.zeros((len(a),2));preflop=0
 for (table,),q in d.group_by('table_id'):
  local=z.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(local.select('hand_id').unique(),on='hand_id',how='semi');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(local.select('hand_id').unique(),on='hand_id',how='semi');holes={(hid,p):(CARD[c1],CARD[c2]) for hid,p,c1,c2 in s.select('hand_id','player_id','hole_card_1','hole_card_2').iter_rows()};boards={hid:[CARD[c] for c in b.split()] for hid,b in h.select('hand_id','board_cards').iter_rows()}
  for r in local.to_dicts():
   actor=r['player_1'] if r['actor']==0 else r['player_2'];partner=r['player_2'] if r['actor']==0 else r['player_1'];st=int(r['street_no']);nb=0 if st==0 else st+2
   for j,p in enumerate([actor,partner]):
    cs=holes[r['hand_id'],p];i=r['action_row']
    if st==0:values[i,j]=PF[cs[0]//4,cs[1]//4,int(cs[0]%4==cs[1]%4)];preflop+=1
    else:states.append(canonical(list(cs)+boards[r['hand_id']][:nb]+[-1]*(5-nb)));where.append((i,j))
 unique,inverse=np.unique(np.array(states,np.int8),axis=0,return_inverse=True);lib=ctypes.CDLL(str((ROOT/'precision.dylib').resolve()));lib.poker_precise_equity.argtypes=[ctypes.POINTER(ctypes.c_int8),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)];out=np.empty(len(unique),np.float32);lib.poker_precise_equity(unique.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),len(unique),4096,out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)));assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
 for (i,j),v in zip(where,out[inverse]):values[i,j]=v
 return values,{'action_pairs':len(a),'preflop_preserved_values':preflop,'unique_postflop_card_states':len(unique),'simulations_flop_turn':4096,'river':'all990 legal opposing holdings'},unique,out
def replace(x,cols,values):
 x=x.copy();own,partner=values.T
 x[:,cols.index('equity')]=own;x[:,cols.index('mw_information_gap')]=x[:,cols.index('mw_own')]-own;x[:,cols.index('bet_equity')]=partner;x[:,cols.index('fold_minus_bet_equity')]=own-partner
 return x
def main():
 ROOT.mkdir(exist_ok=True);base=pl.read_parquet(BASE/'event_oof.parquet');parts=[];audit=[];start=time.time();config={'method':__doc__,'baseline':str(BASE),'modified_columns':['equity','mw_information_gap','bet_equity','fold_minus_bet_equity'],'simulation':'4096 flop/turn, exact river, old12000 preflop table retained','models':'same Cat400 depth5 lr.035 L2 8 original primary seeds','scope':'directed/soft primary only'};(ROOT/'config.json').write_text(json.dumps(config,indent=2))
 for family in ['directed_transfer','soft_play']:
  d,a,x,cols,ac,hc,ec=family_data(family);values,info,states,eq=compute(d,a);root=ROOT/family;root.mkdir(exist_ok=True);np.savez_compressed(root/'equity.npz',values=values,states=states,equity=eq);info['mean_absolute_equity_change']=float(np.mean(abs(values[:,0]-x[:,cols.index('equity')])));x=replace(x,cols,values);g=a['row'].to_numpy();r=a['actor'].to_numpy();pp=np.zeros((len(d),2))
  for f in range(4):
   if family=='directed_transfer':y,tr,va=labels(d,a,f)
   else:y0,e,_=target(d,f);y=y0[g];tr=e[g];va=d['fold'].to_numpy()[g]==f
   assert not (tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(root/f'primary_fold{f}.cbm'));pp[g[va],r[va]]=m.predict_proba(x[va],thread_count=2)[:,1]
  if family=='directed_transfer':dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy()
  else:dw=np.ones_like(pp)
  parts.append(d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',(pp*dw).sum(1))));audit.append({'family':family,**info});print('precision',family,info,round(time.time()-start,1),flush=True)
 base.join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary')).drop('new_primary').write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':main()
