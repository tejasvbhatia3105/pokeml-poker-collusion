\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostRegressor,CatBoostClassifier
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session14_outcomes');P=Path('artifacts/policy');C=pl.col
BASE=json.load(open(P/'feature_columns.json'));PARTNER=['equity','made_category','made_kicker','rank_high','rank_low','suited','pocket','hole_suit_matches','hole_board_matches']
EXTRA=['partner_'+c for c in PARTNER]+['facing_partner','partner_start_bb','own_committed_ratio','partner_start_ratio','active_equity_sum','active_equity_max','outside_equity_mean','action_class','log_amount_bb','log_bet_ratio']
COLS=BASE+EXTRA
CONFIG={'training_sample_per_table_street':[180,80,40,40],'random_partner':'one uniformly sampled currently active other player per sampled action; fixed seed; selection never uses labels','training':'four whole-pool held-out MultiRMSE models, 400 trees depth6 lr .05 L2 20; no early stopping','targets':'actor and partner realized net big blinds divided by 1 + current pot big blinds, clipped [-10,10]','features':'decision-time poker state, actual private-card summaries for retrospective detection, chosen action and sizing; no identities, phase time, future board or outcomes','inference':'actual action and legal fold/check/call/three raise sizes; alternatives require ordinary action propensity >= .03; own fold payoff fixed by already committed chips','interpretation':'observational outcome contrasts; unmeasured response/selection confounding and support limits remain'}
def table_data(table,query=None):
 a=pl.read_parquet(P/'actions'/f'{table}.parquet').filter(C('phase')=='development').with_columns(C('street_no').cast(pl.Int64)).sort('hand_id','action_no','player_id')
 if query is None:
  pieces=[]
  for street,n in enumerate(CONFIG['training_sample_per_table_street']):
   z=a.filter(C('street_no')==street)
   if len(z):pieces.append(z.sample(n=min(n,len(z)),seed=1414+street))
  a=pl.concat(pieces).sort('hand_id','action_no','player_id')
 else:a=a.join(query.select('hand_id').unique(),on='hand_id',how='semi')
 a=a.with_row_index('action_index');st=pl.read_parquet(P/'states'/f'{table}.parquet');active=a.select('action_index','hand_id','street_no','action_no').join(st.select('hand_id','street_no','player_id','fold_no','equity'),on=['hand_id','street_no']).filter(C('fold_no')>=C('action_no'));stats=active.group_by('action_index').agg(C('equity').sum().alias('active_equity_sum'),C('equity').max().alias('active_equity_max'))
 if query is None:
  z=a.join(st.select('hand_id','street_no',C('player_id').alias('partner'),C('fold_no').alias('partner_fold'),*[C(c).alias('partner_'+c) for c in PARTNER]),on=['hand_id','street_no']).filter((C('partner')!=C('player_id'))&(C('partner_fold')>=C('action_no'))).sort('action_index','partner');z=z.with_columns(pl.struct('hand_id','player_id','action_no','partner').hash(seed=1414).alias('pick')).sort('action_index','pick').unique('action_index',keep='first',maintain_order=True).drop('pick')
 else:
  mapping=pl.concat([query.select('pair_id','hand_id',C('player_1').alias('player_id'),C('player_2').alias('partner')),query.select('pair_id','hand_id',C('player_2').alias('player_id'),C('player_1').alias('partner'))]);z=a.join(mapping,on=['hand_id','player_id']).join(st.select('hand_id','street_no',C('player_id').alias('partner'),C('fold_no').alias('partner_fold'),*[C(c).alias('partner_'+c) for c in PARTNER]),on=['hand_id','street_no','partner']);z=z.filter(C('partner_fold')>=C('action_no'))
 seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').select('hand_id','player_id','starting_stack','net_chips');z=z.join(stats,on='action_index').join(seats.rename({'starting_stack':'own_start','net_chips':'own_net'}),on=['hand_id','player_id']).join(seats.rename({'player_id':'partner','starting_stack':'partner_start','net_chips':'partner_net'}),on=['hand_id','partner']);z=z.with_columns((C('last_aggressor')==C('partner')).fill_null(False).cast(pl.Float32).alias('facing_partner'),(C('partner_start')/C('big_blind')).alias('partner_start_bb'),((C('own_start')/C('big_blind')-C('stack_bb'))/(1+C('pot_bb'))).alias('own_committed_ratio'),(C('partner_start')/C('big_blind')/(1+C('pot_bb'))).alias('partner_start_ratio'),((C('active_equity_sum')-C('equity')-C('partner_equity'))/(C('players_active')-2).clip(1)).alias('outside_equity_mean'),(C('own_net')/C('big_blind')/(1+C('pot_bb'))).clip(-10,10).alias('target_own'),(C('partner_net')/C('big_blind')/(1+C('pot_bb'))).clip(-10,10).alias('target_partner'));assert np.isfinite(z.select(COLS+['target_own','target_partner']).to_numpy()).all();return z

def prepare():
 (ROOT/'sample').mkdir(parents=True,exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));(ROOT/'columns.json').write_text(json.dumps(COLS,indent=2));folds=json.load(open(P/'table_folds.json'));start=time.time()
 for i,(table,f) in enumerate(sorted(folds.items())):
  path=ROOT/'sample'/f'{table}.parquet'
  if path.exists():continue
  z=table_data(table);z.select('hand_id','player_id','partner',*COLS,'target_own','target_partner').with_columns(pl.lit(table).alias('table_id'),pl.lit(f).alias('fold'),pl.selectors.numeric().cast(pl.Float32)).write_parquet(path,compression='zstd')
  if i%50==0:print('outcome sample',i,round(time.time()-start,1),flush=True)

def train():
 d=pl.read_parquet(list((ROOT/'sample').glob('T*.parquet')));assert d['table_id'].n_unique()==400;X=d.select(COLS).to_numpy();y=d.select('target_own','target_partner').to_numpy();fv=d['fold'].to_numpy();audit=[];start=time.time()
 for f in range(4):
  tr=fv!=f;va=~tr;path=ROOT/f'outcome_fold{f}.cbm';m=CatBoostRegressor(iterations=400,depth=6,learning_rate=.05,l2_leaf_reg=20,loss_function='MultiRMSE',thread_count=3,random_seed=14140+f,verbose=False,allow_writing_files=False)
  if path.exists():m.load_model(str(path))
  else:m.fit(X[tr],y[tr]);m.save_model(str(path))
  pred=m.predict(X[va],thread_count=3);err=((pred-y[va])**2).mean(0);null=((y[tr].mean(0)-y[va])**2).mean(0);audit.append({'fold':f,'training_rows':int(tr.sum()),'validation_rows':int(va.sum()),'mse_actor_partner':err.tolist(),'training_mean_predictor_mse':null.tolist(),'competition_labels_used':False});print('outcome model',f,round(time.time()-start,1),audit[-1],flush=True)
 (ROOT/'outcome_audit.json').write_text(json.dumps(audit,indent=2))

def action_contrasts(z,m,policy):
 x=z.select(COLS).to_numpy();n=len(z);call=z['call_bb'].to_numpy();stack=z['stack_bb'].to_numpy();pot=z['pot_bb'].to_numpy();actual=z['action_class'].to_numpy().astype(int);prob=policy.predict_proba(z.select(BASE).to_numpy(),thread_count=2);legal=np.ones_like(prob,bool);legal[call>0,1]=False;legal[call<=0,0]=False;legal[call<=0,2]=False;legal[stack<=call,3]=False;prob*=legal;prob/=prob.sum(1)[:,None];assert legal[np.arange(n),actual].all();choices=[x.copy()];valid=[np.ones(n,bool)];classes=[actual];classcol=COLS.index('action_class');amountcol=COLS.index('log_amount_bb');ratiocol=COLS.index('log_bet_ratio')
 for k,ratio in [(0,0),(1,0),(2,0),(3,.33),(3,.75),(3,1.5)]:
  v=x.copy();amount=np.zeros(n) if k<2 else np.minimum(call,stack) if k==2 else np.minimum(stack,np.maximum(2*call,np.maximum(1,pot*ratio)));v[:,classcol]=k;v[:,amountcol]=np.log1p(amount);v[:,ratiocol]=np.log(amount/np.maximum(pot,1e-6)+.01);choices.append(v);valid.append(legal[:,k]&(prob[:,k]>=.03)&((amount>call) if k==3 else True));classes.append(np.full(n,k))
 q=np.stack([m.predict(v,thread_count=2) for v in choices],1);valid=np.stack(valid,1);cl=np.stack(classes,1);q[:,:,0]=np.where(cl==0,-z['own_committed_ratio'].to_numpy()[:,None],q[:,:,0]);best=np.argmax(np.where(valid,q[:,:,0],-1e9),1);rows=np.arange(n);own=q[:,0,0];partner=q[:,0,1];selfish=q[rows,best];team=q.sum(2);team_best=np.max(np.where(valid,team,-1e9),1);fields={'own_regret':np.maximum(0,selfish[:,0]-own),'partner_gain':partner-selfish[:,1],'team_regret':np.maximum(0,team_best-team[:,0]),'observed_own_prediction':own,'observed_partner_prediction':partner,'factual_own_residual':z['target_own'].to_numpy()-own,'factual_partner_residual':z['target_partner'].to_numpy()-partner};fields['transfer_contrast']=fields['own_regret']*np.maximum(0,fields['partner_gain']);fields['sacrifice_without_team_loss']=fields['own_regret']/(1+fields['team_regret']);assert np.isfinite(np.column_stack(list(fields.values()))).all();return fields

def hand_features(table,g,model,policy):
 z=table_data(table,g);fields=action_contrasts(z,model,policy);z=z.with_columns(*[pl.Series(k,v) for k,v in fields.items()]);expr=[]
 for role,rm in [('lower',C('own_net')<=C('partner_net')),('higher',C('own_net')>=C('partner_net'))]:
  for context,cm in [('partner',C('facing_partner')>0),('outside',(C('facing_partner')==0)&(C('players_active')>=3)),('hu',C('players_active')==2)]:
   for name in fields:
    v=C(name).filter(rm&cm);prefix=f'contrast_{role}_{context}_{name}';expr.extend([v.sum().alias(prefix+'_sum'),v.max().fill_null(0).alias(prefix+'_max')])
 out=z.group_by('pair_id','hand_id').agg(expr);return g.select('pair_id','hand_id').join(out,on=['pair_id','hand_id'],how='left',validate='1:1').fill_null(0).with_columns(pl.selectors.numeric().cast(pl.Float32))

def build_features():
 (ROOT/'features').mkdir(exist_ok=True);d=hand_data();players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');query=d.select('pair_id','hand_id','table_id').join(players,on='pair_id',validate='m:1');folds=json.load(open(P/'table_folds.json'));models=[];policies=[]
 for f in range(4):
  m=CatBoostRegressor();m.load_model(str(ROOT/f'outcome_fold{f}.cbm'));models.append(m);p=CatBoostClassifier();p.load_model(str(P/f'action_fold{f}.cbm'));policies.append(p)
 start=time.time()
 for i,((table,),g) in enumerate(query.group_by('table_id')):
  path=ROOT/'features'/f'{table}.parquet'
  if path.exists():continue
  f=folds[table];out=hand_features(table,g.drop('table_id'),models[f],policies[f]);out.write_parquet(path,compression='zstd')
  if i%50==0:print('outcome features',i,round(time.time()-start,1),flush=True)
 out=pl.read_parquet(list((ROOT/'features').glob('T*.parquet')));assert len(out)==len(d);out.write_parquet(ROOT/'hand_features.parquet')
if __name__=='__main__':prepare();train();build_features()
