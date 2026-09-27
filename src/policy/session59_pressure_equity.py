\
\
\
\
\
\
\
import ctypes,itertools,json,time
from pathlib import Path
import numpy as np,polars as pl
from cards import CARD
import session58_pressure_comparison as training
from session5_multiway_features import LIB
ROOT=Path('artifacts/evidence_session59_pressure_equity');C=pl.col
COLS=['precise_mw_own','precise_mw_partner','precise_mw_partner_fold_gain','precise_mw_team','precise_mw_call_edge','precise_mw_information_gap','precise_mw_fold_value']
PERMS=list(itertools.permutations(range(4)))
def canonical(cs):
 candidates=[]
 for p in PERMS:
  v=[-1 if c<0 else 4*(c//4)+p[c%4] for c in cs];h=sum((sorted(v[2*j:2*j+2]) for j in range(6)),[]);b=sorted(c for c in v[12:] if c>=0);candidates.append(tuple(h+b+[-1]*(5-len(b))))
 return min(candidates)
def equity(states,masks,sims):
 states=np.ascontiguousarray(states,dtype=np.int8);masks=np.ascontiguousarray(masks,dtype=np.uint8);out=np.empty((len(states),6),np.float32);LIB.poker_multiway(states.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),len(states),sims,out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)));assert np.isfinite(out).all() and np.allclose(out.sum(1),1,atol=1e-6);return out
def states(d,a,players=None):
 people=(pl.read_csv('data/development_labels.csv') if players is None else players).select('pair_id','player_1','player_2');z=a.drop('action_row',strict=False).with_row_index('action_row');ss=[];mm=[];metadata=[]
 for (table,),q in d.group_by('table_id'):
  local=z.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');need=local.select('hand_id').unique();s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(need,on='hand_id',how='semi').sort('hand_id','seat_no');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').join(need,on='hand_id',how='semi');local=local.join(raw.select('hand_id','street_no','action_no','player_id'),on=['hand_id','street_no','action_no'],validate='m:1').join(people,on='pair_id',validate='m:1');boards={hid:[CARD[c] for c in b.split()] for hid,b in h.select('hand_id','board_cards').iter_rows()};rosters={};holes={}
  for (hid,),p in s.group_by('hand_id'):
   rosters[hid]=p['player_id'].to_list();holes[hid]=sum(([CARD[c1],CARD[c2]] for c1,c2 in p.select('hole_card_1','hole_card_2').iter_rows()),[])
  folds={(hid,p):int(n) for hid,p,n in raw.filter(C('action_class')==0).select('hand_id','player_id','action_no').iter_rows()}
  for r in local.to_dicts():
   hid=r['hand_id'];own=r['player_id'];partner=r['player_2'] if own==r['player_1'] else r['player_1'];assert own in [r['player_1'],r['player_2']];pa=rosters[hid].index(own);pb=rosters[hid].index(partner);mask=sum(1<<j for j,p in enumerate(rosters[hid]) if folds.get((hid,p),999)>=r['action_no']);assert mask&(1<<pa) and mask&(1<<pb) and mask.bit_count()==int(r['players_active']);nb=0 if r['street_no']==0 else int(r['street_no'])+2;cs=holes[hid]+boards[hid][:nb]+[-1]*(5-nb);assert len(cs)==17;offset=len(ss);ss.extend([cs,cs]);mm.extend([mask,mask&~(1<<pa)]);metadata.append([r['action_row'],offset,offset+1,pa,pb])
 return np.array(ss,np.int8),np.array(mm,np.uint8),np.array(metadata,np.int64)
def fields(eq,meta,a):
 v=np.zeros((len(a),7),np.float32);odds=a['pot_odds'].to_numpy();ordinary=a['equity'].to_numpy();pot=a['pot_bb'].to_numpy()
 for row,before,after,own,partner in meta:
  x,y=eq[before,own],eq[before,partner];gain=eq[after,partner]-y;v[row]=[x,y,gain,x+y,x-float(odds[row]),x-float(ordinary[row]),gain*np.log1p(float(pot[row]))]
 return v
def compute(d,a,players=None):
 ROOT.mkdir(exist_ok=True);start=time.time();ss,mm,meta=states(d,a,players);old=equity(ss,mm,128);baseline=fields(old,meta,a);oldcols=[s.removeprefix('precise_') for s in COLS];rawerror=float(abs(baseline-a.select(oldcols).to_numpy()).max());assert rawerror<1e-6
 canonical_states=np.array([canonical(cs) for cs in ss],np.int8);packed=np.column_stack([canonical_states,mm]);unique,inverse=np.unique(packed,axis=0,return_inverse=True);print('precision states',len(unique),'raw_error',rawerror,'seconds',round(time.time()-start,1),flush=True);out=equity(unique[:,:17],unique[:,17],4096);v=fields(out[inverse],meta,a);np.savez_compressed(ROOT/'equity_audit.npz',states=ss,masks=mm,meta=meta,canonical_unique=unique,inverse=inverse,precise_equity=out,raw_equity=old);audit={'actions':len(a),'preflop_actions':int((a['street_no']==0).sum()),'unique_active_card_states':len(unique),'simulations_preflop_flop':4096,'turn_river':'exact legal board completions','raw128_feature_replay_error':rawerror,'mean_absolute_own_equity_change':float(abs(v[:,0]-baseline[:,0]).mean()),'feature_runtime_seconds':time.time()-start};print(json.dumps(audit),flush=True);return v,audit
def main():
                                                                      
 training.ROOT=ROOT;training.COLS=COLS;training.compute=compute;training.__doc__=__doc__;training.main()
if __name__=='__main__':main()
