\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session55_current_targets import state
from session37_bet_fold import paired_features
from session35_fold_likelihood import labels
from session29_shared_fold_witness import target
from session57_isolation_pressure import data as pressure_data,noisy_or
from session8_data import hand_data,targets
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session92_chip_signatures');C=pl.col
RATIOS=[(1,4),(1,3),(1,2),(2,3),(3,4),(1,1),(5,4),(3,2),(2,1),(3,1)]
RAW=['amount','amount_to','pot_before','stack_before','to_call','big_blind']
def formulas(a):
 A,T,P,S,CALL,B=[a[c].to_numpy().astype(np.int64) for c in RAW];assert (A>CALL).all() and (A<=S).all() and (T>=A).all() and (B>0).all();assert max(map(lambda v:int(abs(v).max()),[A,T,P,S,CALL,B]))<10**12;columns=[];values=[]
 for name,x,y in [('paid_pot',A,P),('excess_after_call',A-CALL,P+CALL),('to_pot',T,P),('paid_stack',A,S),('paid_blind',A,B)]:
  for n,d in RATIOS:
   residual=d*x-n*y;error=residual/(d*B);fields=[np.sign(error)*np.log1p(abs(error)),residual==0,(residual<=0)&(residual>-d)];columns.extend([f'{name}_{n}_{d}_{k}' for k in ['signed_log_error_bb','exact','floor']]);values.extend(fields)
 columns.append('street_committed_log_bb');values.append(np.log1p((T-A)/B));return np.column_stack(values).astype(np.float32),columns
def raw_queries(q):
 pieces=[]
 for (table,),g in q.group_by('table_id'):
  a=pl.read_parquet(f'artifacts/compact/actions/table_id={table}/*.parquet').select('hand_id','action_no','player_id',*RAW[:-1]);h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').select('hand_id','big_blind');pieces.append(g.join(a,on=['hand_id','action_no'],validate='m:1').join(h,on='hand_id',validate='m:1'))
 z=pl.concat(pieces).sort('action_row');np.testing.assert_array_equal(z['action_row'].to_numpy(),np.arange(len(q)));assert z.select(RAW).null_count().to_numpy().sum()==0;return z
def prepare():
 ROOT.mkdir(exist_ok=True);states=state();ledger=pl.read_parquet('artifacts/evidence_session8/ledger_actions.parquet').with_columns(C('action_no').cast(pl.Int64));lc=[c for c in ledger.columns if c.startswith('ledger_')];reports=[]
 for fam in ['directed_transfer','soft_play','coordinated_isolation']:
  if fam=='coordinated_isolation':_,d,a,ac=pressure_data();q=a.with_row_index('action_row').select('action_row','pair_id','hand_id',C('action_no').cast(pl.Int64)).join(d.select('pair_id','hand_id','table_id'),on=['pair_id','hand_id'],validate='m:1')
  else:
   v=states[fam];d,a=v['d'],v['a'];_,alignment=paired_features(d,a);q=alignment.select('action_row','pair_id','hand_id',C('bet_action_no').alias('action_no'),'partner').join(d.select('pair_id','hand_id','table_id'),on=['pair_id','hand_id'],validate='m:1')
  raw=raw_queries(q)
  if fam!='coordinated_isolation':assert (raw['player_id']==raw['partner']).all()
  else:
   people=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');z=raw.join(people,on='pair_id',validate='m:1');assert ((z['player_id']==z['player_1'])|(z['player_id']==z['player_2'])).all()
  x,cols=formulas(raw);ll=raw.select('hand_id','action_no').join(ledger,on=['hand_id','action_no'],validate='m:1',maintain_order='left').select(lc);assert ll.null_count().to_numpy().sum()==0;x=np.column_stack([x,ll.to_numpy()]).astype(np.float32);raw.write_parquet(ROOT/f'{fam}_raw.parquet');np.savez_compressed(ROOT/f'{fam}_features.npz',x=x);reports.append({'family':fam,'actions':len(raw),'formula_fields':len(cols),'ledger_fields':len(lc)});print(reports[-1],flush=True)
 (ROOT/'columns.json').write_text(json.dumps({'formula':cols,'ledger':lc,'all':cols+lc},indent=2));(ROOT/'input_audit.json').write_text(json.dumps(reports,indent=2))
def fit(x,y,tr,f,head,path,loss='Logloss'):
 m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function=loss,random_seed=6311+11*f+head,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(path));return m
def train():
 (ROOT/'config.json').write_text(json.dumps({'method':__doc__,'ratios':RATIOS,'schedule':'original Cat400 D5 lr.035 L2=8 head seeds; original3 EM isolation rounds','heads':'direct/soft primary and both isolation heads','targets':'unchanged R32/R33 public targets, censoring and donor weights','features':'151 exact chip arithmetic fields + 25 original session8 ledger fields at matched action; fixed original base inputs','no_family_or_weight_selection':True},indent=2));states=state();full=hand_data();base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');parts=[];audit=[];start=time.time()
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];extra=np.load(ROOT/f'{fam}_features.npz')['x'];x=np.column_stack([v['x'],extra]);dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2));pp=np.zeros((len(d),2))
  for f in range(4):
   if fam=='directed_transfer':y,tr,va=labels(d,a,f)
   else:yy,e,_=target(d,f);y=yy[g];tr=e[g];va=fv[g]==f
   assert not(tr&va).any();m=fit(x,y,tr,f,0,ROOT/f'{fam}_primary_fold{f}.cbm');pp[g[va],actor[va]]=m.predict_proba(x[va],thread_count=2)[:,1];audit.append({'family':fam,'head':1,'fold':f,'training_actions':int(tr.sum()),'positive_actions':int(y[tr].sum()),'validation_overlap':0});print('chip heads',fam,f,round(time.time()-start,1),flush=True)
  parts.append(d.select('pair_id','hand_id').with_columns(pl.Series('replace_primary',(pp*dw).sum(1))))
 _,d,a,ac=pressure_data();g=a['row'].to_numpy();fv=d['fold'].to_numpy();cnt=np.bincount(g,minlength=len(d));hc=json.load(open('artifacts/evidence_session59_pressure_equity/config.json'))['hand_columns'];x=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x'],np.load(ROOT/'coordinated_isolation_features.npz')['x']]);fm=full['behavior_family'].to_numpy()=='coordinated_isolation';pred=np.zeros((len(d),2))
 for f in range(4):
  p1,p2,e1,e2,_=targets(full,f,'coordinated_isolation');ys=np.column_stack([p1,p2])[fm];es=np.column_stack([e1,e2])[fm];va=fv[g]==f
  for head in range(2):
   y=ys[:,head];tr=es[g,head];assert not(tr&va).any();gt=g[tr];yt=y[gt];resp=yt/cnt[gt]
   for em in range(3):
    yy=np.zeros(len(a));yy[tr]=resp;m=fit(x,yy,tr,f,head,ROOT/f'isolation_head{head+1}_fold{f}_em{em}.cbm','CrossEntropy');p=m.predict_proba(x[tr],thread_count=2)[:,1];hp=noisy_or(p,gt,len(d));resp=np.where(yt,p/np.maximum(hp[gt],1e-8),0).clip(0,1)
   hp=noisy_or(m.predict_proba(x[va],thread_count=2)[:,1],g[va],len(d));pred[fv==f,head]=hp[fv==f];audit.append({'family':'coordinated_isolation','head':head+1,'fold':f,'training_actions':int(tr.sum()),'validation_overlap':0})
  print('chip heads isolation',f,round(time.time()-start,1),flush=True)
 iso=d.select('pair_id','hand_id').with_columns(pl.Series('replace_primary',pred[:,0]),pl.Series('replace_secondary',pred[:,1]));q=base.join(pl.concat(parts+[iso.select('pair_id','hand_id','replace_primary')]),on=['pair_id','hand_id'],validate='1:1').join(iso.select('pair_id','hand_id','replace_secondary'),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(C('replace_primary').alias('bg_primary'),pl.coalesce('replace_secondary','bg_secondary').alias('bg_secondary')).drop('replace_primary','replace_secondary');assert len(q)==len(base);q.write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 else:train()
