\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session15_responses');P=Path('artifacts/policy');C=pl.col
CONFIG={'representation':'Actions of players outside the focal pair, while both members remain active; actual outsider cards and ordinary-policy surprises','contexts':['both_alive','facing_pair','both_raised','facing_lower','facing_higher'],'chronology':'sort pair/hand/street/action before prior raise counts, exclude current action from history','leakage':'no evidence labels, types, IDs or future fold status as model features; fold index only routes ordinary policy','controls':'all features versus both_alive response summaries alone'}
def build(table,query,policy,size,cols):
 a=pl.read_parquet(P/'actions'/f'{table}.parquet').join(query.select('hand_id').unique(),on='hand_id',how='semi').with_columns(C('street_no').cast(pl.Int64));X=a.select(cols).to_numpy();p=policy.predict_proba(X,thread_count=2);sizehat=size.predict(X,thread_count=2);act=a['action_class'].to_numpy().astype(int);call=a['to_call'].to_numpy()>0;legal=np.ones_like(p,bool);legal[call,1]=False;legal[~call,0]=False;legal[~call,2]=False;p*=legal;p/=p.sum(1)[:,None];assert legal[np.arange(len(act)),act].all();a=a.with_columns(*[pl.Series(f'p{k}',p[:,k]) for k in range(4)],pl.Series('size_residual',a['log_bet_ratio'].to_numpy()-sizehat));st=pl.read_parquet(P/'states'/f'{table}.parquet').select('hand_id','player_id','street_no','fold_no');seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').select('hand_id','player_id','net_chips');q=query.join(seats.rename({'player_id':'player_1','net_chips':'net1'}),on=['hand_id','player_1']).join(seats.rename({'player_id':'player_2','net_chips':'net2'}),on=['hand_id','player_2']);z=a.join(q,on='hand_id').join(st.rename({'player_id':'player_1','fold_no':'fold1'}),on=['hand_id','player_1','street_no']).join(st.rename({'player_id':'player_2','fold_no':'fold2'}),on=['hand_id','player_2','street_no']);keys=['pair_id','hand_id','street_no'];z=z.sort(*keys,'action_no','player_id');z=z.with_columns(*[((C('player_id')==C(f'player_{k}'))&(C('action_class')==3)).cast(pl.Int32).cum_sum().shift(1).over(keys).fill_null(0).alias(f'prior_raise{k}') for k in [1,2]]);z=z.filter((C('player_id')!=C('player_1'))&(C('player_id')!=C('player_2'))&(C('fold1')>=C('action_no'))&(C('fold2')>=C('action_no')))
 f1=(C('last_aggressor')==C('player_1')).fill_null(False);f2=(C('last_aggressor')==C('player_2')).fill_null(False);ctx={'both_alive':pl.lit(True),'facing_pair':f1|f2,'both_raised':(C('prior_raise1')>0)&(C('prior_raise2')>0),'facing_lower':(f1&(C('net1')<=C('net2')))|(f2&(C('net2')<=C('net1'))),'facing_higher':(f1&(C('net1')>=C('net2')))|(f2&(C('net2')>=C('net1')))};expr=[]
 for context,mask in ctx.items():
  for k in range(4):
   pk=C(f'p{k}');isaction=C('action_class')==k;gate=mask&isaction;prefix=f'response_{context}_{k}';expr.extend([gate.sum().alias(prefix+'_count'),(isaction.cast(pl.Float64)-pk).filter(mask).sum().alias(prefix+'_residual'),(pk*(1-pk)).filter(mask).sum().alias(prefix+'_variance'),(-pk.clip(1e-7,1).log()).filter(gate).max().fill_null(0).alias(prefix+'_surprise'),C('equity').filter(gate).mean().fill_null(-1).alias(prefix+'_equity_mean'),C('equity').filter(gate).max().fill_null(-1).alias(prefix+'_equity_max'),C('call_stack').filter(gate).max().fill_null(0).alias(prefix+'_call_stack_max'),C('pot_odds').filter(gate).mean().fill_null(0).alias(prefix+'_pot_odds_mean')])
  raisegate=mask&(C('action_class')==3);expr.extend([C('size_residual').filter(raisegate).sum().alias(f'response_{context}_size_sum'),C('size_residual').abs().filter(raisegate).max().fill_null(0).alias(f'response_{context}_size_absmax')])
 out=z.group_by('pair_id','hand_id').agg(expr);out=query.select('pair_id','hand_id').join(out,on=['pair_id','hand_id'],how='left',validate='1:1').fill_null(0).with_columns(pl.selectors.numeric().cast(pl.Float32));return out
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();q=d.select('pair_id','hand_id','table_id').join(pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),on='pair_id',validate='m:1');cols=json.load(open(P/'feature_columns.json'));folds=json.load(open(P/'table_folds.json'));models=[];sizes=[];start=time.time();swap=None
 for f in range(4):
  m=CatBoostClassifier();m.load_model(str(P/f'action_fold{f}.cbm'));models.append(m);s=CatBoostRegressor();s.load_model(str(P/f'size_fold{f}.cbm'));sizes.append(s)
 for i,((table,),g) in enumerate(q.group_by('table_id')):
  path=ROOT/f'{table}.parquet';f=folds[table]
  if path.exists():continue
  out=build(table,g.drop('table_id'),models[f],sizes[f],cols);assert np.isfinite(out.select(pl.selectors.numeric()).to_numpy()).all();out.write_parquet(path,compression='zstd')
  if swap is None:
   v=build(table,g.drop('table_id').rename({'player_1':'player_2','player_2':'player_1'}),models[f],sizes[f],cols).sort('pair_id','hand_id');orig=out.sort('pair_id','hand_id');swap=float(abs(v.select(pl.selectors.numeric()).to_numpy()-orig.select(pl.selectors.numeric()).to_numpy()).max());assert swap<1e-5;(ROOT/'feature_audit.json').write_text(json.dumps({'endpoint_swap_max_error':swap,'first_table':table},indent=2))
  if i%50==0:print('response features',i,round(time.time()-start,1),flush=True)
 out=pl.read_parquet(list(ROOT.glob('T*.parquet')));assert len(out)==len(d);out.write_parquet(ROOT/'hand_features.parquet')
if __name__=='__main__':main()
