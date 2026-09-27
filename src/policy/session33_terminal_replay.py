\
\
\
\
import os,json,ctypes,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import hand_data
C=pl.col;ROOT=Path('artifacts/evidence_session33_rollout')
LIB=ctypes.CDLL(str(Path('artifacts/evidence_session5/multiway.dylib').resolve()));LIB.poker_rank.argtypes=[ctypes.POINTER(ctypes.c_int8),ctypes.c_int];LIB.poker_rank.restype=ctypes.c_uint32
CARD={r+s:4*i+j for i,r in enumerate('23456789TJQKA') for j,s in enumerate('cdhs')}
def ranks(seats,board):
 out=[]
 for a,b in seats.select('hole_card_1','hole_card_2').iter_rows():
  c=np.array([CARD[a],CARD[b]]+[CARD[c] for c in board.split()],np.int8);out.append(LIB.poker_rank(c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),len(c)))
 return np.array(out)
def payout(contrib,alive,rank,sidepots=True):
 result=np.zeros(len(contrib));levels=np.unique(contrib[contrib>0]) if sidepots else np.array([1.]);previous=0.
 for level in levels:
  present=contrib>=level if sidepots else np.ones(len(contrib),bool);pot=(level-previous)*present.sum() if sidepots else contrib.sum();eligible=present&alive
  if not eligible.any():return None
  winners=eligible&(rank==rank[eligible].max());result[winners]+=pot/winners.sum();previous=level
 return result
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();records=[];examples=[];start=time.time()
 for (table,),q in d.group_by('table_id'):
  need=q.select('hand_id').unique();h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');meta={r['hand_id']:r for r in h.to_dicts()}
  for (hid,),z in s.group_by('hand_id'):
   z=z.sort('seat_no');hh=meta[hid];con=z['total_contribution'].to_numpy().astype(float);actual=con+z['net_chips'].to_numpy();alive=~z['folded'].to_numpy();rank=ranks(z,hh['board_cards']);single=payout(con,alive,rank,False);side=payout(con,alive,rank,True);se=float(abs(single-actual).max()) if single is not None else None;pe=float(abs(side-actual).max()) if side is not None else None;row={'hand_id':hid,'table_id':table,'active':int(alive.sum()),'allin_layers':len(np.unique(con[con>0])),'pot_error':float(con.sum()-hh['final_pot']),'net_sum':int(z['net_chips'].sum()),'single_max_error':se,'side_max_error':pe};records.append(row)
   if (pe is None or pe>2) and len(examples)<40:examples.append({'hand':hh,'seats':z.to_dicts(),'ranks':rank.tolist(),'single':None if single is None else single.tolist(),'side':None if side is None else side.tolist()})
 out=pl.DataFrame(records);out.write_parquet(ROOT/'terminal_audit.parquet');report={'purpose':__doc__,'hands':len(out),'multiple_active_hands':int((out['active']>1).sum()),'nonzero_pot_errors':int((out['pot_error']!=0).sum()),'nonzero_net_sums':int((out['net_sum']!=0).sum()),'single_error_above2':int((out['single_max_error']>2).sum()),'side_error_above2':int((out['side_max_error']>2).sum()),'undefined_side_payout':out['side_max_error'].null_count(),'single_max_error':out['single_max_error'].max(),'side_max_error':out['side_max_error'].max(),'seconds':time.time()-start};(ROOT/'terminal_report.json').write_text(json.dumps(report,indent=2));(ROOT/'terminal_examples.json').write_text(json.dumps(examples,indent=2,default=str));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
