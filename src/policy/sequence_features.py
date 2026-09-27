import polars as pl
C=pl.col
def augment(d):
 d=d.sort('pair_id','phase','time')
 gates={
 'both_preflop_raise':(C('weak_preflop_0_action_code')>=3)&(C('strong_preflop_0_action_code')>=3),
 'early_both_raise':(C('weak_preflop_0_action_code')>=3)&(C('strong_preflop_0_action_code')>=3)&(C('weak_preflop_0_players_active')>=4)&(C('strong_preflop_0_players_active')>=4),
 'preflop_pressure':C('preflop_agg_out')>=2,
 'weak_call':C('weak_call_partner')>0,
 'partner_fold':C('strong_fold_partner')>0,
 'partner_check':C('strong_check_hu')>0,
 'no_conflict':(C('face_act')>0)&(C('agg_alive')==0),
 }
 expr=[];cols=[]
 for name,gate in gates.items():
  name='preceding_'+name;cols.append(name)
  expr.append((gate.cast(pl.Int32).cum_sum().over(['pair_id','phase'])-gate.cast(pl.Int32)).alias(name))
 return d.with_columns(expr),cols
