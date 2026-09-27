\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,time,numpy as np,polars as pl
from catboost import CatBoostClassifier
from cards import CARD
from pair_equity import pair_equity
C=pl.col;root=Path('artifacts/policy');dest=root/'value_pairs';hdest=root/'value_hands';dest.mkdir(exist_ok=True);hdest.mkdir(exist_ok=True);cols=json.loads((root/'feature_columns.json').read_text());folds=json.loads((root/'table_folds.json').read_text());models=[]
for f in range(4):
 m=CatBoostClassifier();m.load_model(str(root/f'action_fold{f}.cbm'));models.append(m)
t=time.time()
for ti,path in enumerate(sorted((root/'actions').glob('*.parquet'))):
 out=dest/path.name
 if out.exists():continue
 table=path.stem;a=pl.read_parquet(path);X=a.select(cols).to_numpy();dev=a['phase'].to_numpy()=='development';p=np.zeros((len(a),4))
 for mask,fs in [(dev,[folds[table]]),(~dev,range(4))]:p[mask]=np.mean([models[f].predict_proba(X[mask],thread_count=4) for f in fs],axis=0)
 call=a['to_call'].to_numpy()>0;legal=np.ones_like(p);legal[call,1]=0;legal[~call,0]=0;legal[~call,2]=0;p*=legal;p/=p.sum(1,keepdims=True);a=a.with_columns(*[pl.Series(f'p{k}',p[:,k]) for k in [0,1,2]],C('street_no').cast(pl.Int64))
 h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet');s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').select('hand_id','player_id','hole_card_1','hole_card_2','net_chips').with_columns(C('hole_card_1').replace_strict(CARD).alias('c1'),C('hole_card_2').replace_strict(CARD).alias('c2'))
 st=pl.read_parquet(root/'states'/path.name).select('hand_id','player_id','street_no','fold_no')
 z=a.join(st.rename({'player_id':'partner','fold_no':'partner_fold'}),on=['hand_id','street_no']).filter(C('partner_fold')>=C('action_no')).with_columns(pl.len().over(['hand_id','action_no']).alias('computed_active')).filter(C('player_id')!=C('partner')).with_columns((C('last_aggressor')==C('partner')).fill_null(False).alias('facing'),(C('computed_active')==2).alias('hu')).filter(C('facing')|C('hu'))
 z=z.with_columns(pl.min_horizontal('player_id','partner').alias('player_1'),pl.max_horizontal('player_id','partner').alias('player_2'))
 query=z.select('hand_id','player_1','player_2','street_no').unique().join(s.select('hand_id',C('player_id').alias('player_1'),C('c1').alias('a1'),C('c2').alias('a2')),on=['hand_id','player_1']).join(s.select('hand_id',C('player_id').alias('player_2'),C('c1').alias('b1'),C('c2').alias('b2')),on=['hand_id','player_2'])
 states=np.full((len(query),9),-1,np.int8);states[:,:4]=query.select('a1','a2','b1','b2').to_numpy();boards={hid:[CARD[c] for c in board.split()] for hid,board in h.select('hand_id','board_cards').iter_rows()}
 for i,(hid,street) in enumerate(query.select('hand_id','street_no').iter_rows()):
  if street:states[i,4:street+6]=boards[hid][:street+2]
 query=query.select('hand_id','player_1','player_2','street_no').with_columns(pl.Series('eq1',pair_equity(states,128)));z=z.join(query,on=['hand_id','player_1','player_2','street_no']).with_columns(pl.when(C('player_id')==C('player_1')).then(C('eq1')).otherwise(1-C('eq1')).alias('pair_eq'))
                                                                       
 z=z.with_columns(pl.min_horizontal(C('call_bb'),C('stack_bb')).alias('cost'),(C('pot_bb')-(C('call_bb')-C('stack_bb')).clip(0)).clip(0).alias('effective_pot'))
 z=z.with_columns((C('pair_eq')*(C('effective_pot')+C('cost'))-C('cost')).alias('call_ev'))
 iscall=(C('action_class')==2)&C('facing');isfold=(C('action_class')==0)&C('facing')&C('hu');ischeck=(C('action_class')==1)&C('hu');loss=(-C('call_ev')).clip(0);foldvalue=C('call_ev').clip(0)
 expr={
 'value_call_loss':pl.when(iscall).then(loss).otherwise(0),
 'value_unexpected_call_loss':pl.when(iscall).then(loss*(1-C('p2'))).otherwise(0),
 'value_hu_call_loss':pl.when(iscall&C('hu')).then(loss).otherwise(0),
 'value_hu_fold_loss':pl.when(isfold).then(foldvalue).otherwise(0),
 'value_unexpected_fold_loss':pl.when(isfold).then(foldvalue*(1-C('p0'))).otherwise(0),
 'value_check_edge':pl.when(ischeck).then((C('pair_eq')-.75).clip(0)*C('effective_pot')*(1-C('p1'))).otherwise(0),
 'value_dominated_call':pl.when(iscall).then((.2-C('pair_eq')).clip(0)*C('cost')).otherwise(0),
 'value_equity_fold':pl.when(isfold).then((C('pair_eq')-.8).clip(0)*C('effective_pot')).otherwise(0),
 }
 z=z.with_columns(*[v.alias(k) for k,v in expr.items()]);vh=z.group_by('hand_id','player_1','player_2').agg(*[C(k).sum() for k in expr])
 roster=s.select('hand_id',C('player_id').alias('player_1'));base=roster.join(roster.rename({'player_1':'player_2'}),on='hand_id').filter(C('player_1')<C('player_2')).join(a.select('hand_id','phase','time_index').unique(),on='hand_id');pairs=pl.read_parquet(root/'pair_features'/path.name).select('player_1','player_2','pair_id').unique();vh=base.join(vh,on=['hand_id','player_1','player_2'],how='left').fill_null(0).join(pairs,on=['player_1','player_2']).sort('pair_id','phase','time_index')
 stats=[]
 for n in expr:
  x=C(n);stats += [x.mean().alias(n+'_mean'),x.max().alias(n+'_max'),x.top_k(3).mean().alias(n+'_top3'),x.sum().alias(n+'_sum'),(x>5).mean().alias(n+'_over5_rate'),(x+1).log().mean().alias(n+'_logmean')]
 f=vh.group_by('pair_id','phase').agg(stats).with_columns(pl.lit(table).alias('table_id'),pl.selectors.float().cast(pl.Float32));f.write_parquet(out,compression='zstd')
 wanted=pl.read_parquet(root/'hand_features'/path.name).select('pair_id','phase').unique();vh.join(wanted,on=['pair_id','phase'],how='semi').with_columns(pl.selectors.float().cast(pl.Float32)).write_parquet(hdest/path.name,compression='zstd')
 if ti%20==0:print(ti,'queries',len(query),'seconds',round(time.time()-t,1),flush=True)
