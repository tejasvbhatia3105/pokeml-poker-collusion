import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl,torch
import session86_action_history as s
C=pl.col
def scalar(rows,index):
 now=rows[index];past=[r for r in rows[:index] if r['hand_id']==now['hand_id']][-s.LENGTH:];z=np.zeros((s.LENGTH,len(s.PUBLIC_NAMES)),np.float32)
 for k,r in enumerate(past):
  fields=[float(r['action_class']==j) for j in range(4)]+[float(r['street_no']==j) for j in range(4)]+[r['log_amount_bb'],np.log1p(max(r['pot_bb'],0)),np.log1p(max(r['call_bb'],0)),np.log1p(max(r['stack_bb'],0)),r['players_active']/6,((r['position']-now['position'])%6)/5,(now['action_no']-r['action_no'])/16]+[r[f'{p}_{j}'] for p in ['style','local_style'] for j in range(4)]+[float(r['player_id']==now['player_id']),float(r['player_id']==now['last_aggressor']),float(r['street_no']==now['street_no'])];z[k]=fields
 return z.astype(np.float16),len(past)
def data():
 saved=np.load(s.ROOT/'sample.npz');meta=pl.read_parquet(s.ROOT/'sample_keys.parquet').with_row_index('cache_row');pc=json.load(open('artifacts/policy/feature_columns.json'));hist=saved['history'];length=saved['length'];checks=mutations=0
 for t,((table,),q) in enumerate(meta.group_by('table_id',maintain_order=True)):
  a=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').filter(C('phase')=='development').sort('hand_id','action_no');ix=q['source_row'].to_numpy();rows=q['cache_row'].to_numpy();h,l=s.history(a,ix);np.testing.assert_array_equal(h,hist[rows]);np.testing.assert_array_equal(l,length[rows]);np.testing.assert_array_equal(a[ix].select(pc).to_numpy(),saved['x'][rows]);np.testing.assert_array_equal(a[ix]['action_class'].to_numpy(),saved['y'][rows]);assert a[ix].select('hand_id','player_id','action_no').equals(q.select('hand_id','player_id','action_no'))
  if t%25==0:
   raw=a.to_dicts()
   for index in ix[np.linspace(0,len(ix)-1,8).astype(int)]:
    hs,ls=scalar(raw,int(index));hv,lv=s.history(a,np.array([index]));np.testing.assert_array_equal(hs,hv[0]);assert ls==lv[0];checks+=1
   index=int(ix[len(ix)//2]);before,_=s.history(a,np.array([index]));future=np.arange(len(a))>=index;mut=a.with_columns(pl.when(pl.Series(future)).then((C('action_class')+1)%4).otherwise(C('action_class')).alias('action_class'),pl.when(pl.Series(future)).then(C('log_amount_bb')+99).otherwise(C('log_amount_bb')).alias('log_amount_bb'),*[pl.lit(-999.).alias(c) for c in ['equity','made_category','made_kicker','rank_high','rank_low','suited','pocket']]);after,_=s.history(mut,np.array([index]));np.testing.assert_array_equal(before,after);mutations+=1
 report={'all_sample_rows_raw_replayed':len(meta),'history_cache_error':0,'independent_scalar_histories':checks,'scalar_error':0,'current_future_action_and_private_card_mutation_checks':mutations,'mutation_error':0,'no_hand_crossings_or_nonpreceding_tokens':True,'history_length_quantiles':np.quantile(length,[0,.25,.5,.75,.95,1]).tolist(),'empty_histories':int((length==0).sum()),'history_columns':s.PUBLIC_NAMES};(s.ROOT/'data_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
def models(folds):
 torch.set_num_threads(3);z=np.load(s.ROOT/'sample.npz');raw=z['x'];h=z['history'];l=z['length'];native=z['fold'];pc=json.load(open('artifacts/policy/feature_columns.json'));a=s.current_transform(raw,pc);call=raw[:,pc.index('call_bb')]>0;count=0
 for f in folds:
  tr=native!=f;va=~tr;inputs=np.load(s.ROOT/f'inputs_fold{f}.npz');np.testing.assert_array_equal(a[tr].mean(0),inputs['mean']);np.testing.assert_array_equal(np.maximum(a[tr].std(0),.01),inputs['scale']);prior=s.reference(raw,native,f);np.testing.assert_array_equal(prior,inputs['prior']);x=((a-inputs['mean'])/inputs['scale']).clip(-20,20).astype(np.float32);vi=np.flatnonzero(va)
  for kind in s.CONFIG['arms']:
   ck=torch.load(s.ROOT/f'{kind}_fold{f}.pt',map_location='cpu',weights_only=False);m=s.Policy(kind,len(pc));m.load_state_dict(ck['state']);p=s.predict(m,x,h,l,prior,call,vi);np.testing.assert_array_equal(p,np.load(s.ROOT/f'{kind}_fold{f}.npy'));rev=s.predict(m,x,h,l,prior,call,vi[:1024][::-1])[::-1];np.testing.assert_allclose(p[:1024],rev,atol=2e-7,rtol=0);count+=1
 report={'folds':folds,'model_replays':count,'saved_probability_error':0,'row_reversal_tolerance':2e-7,'train_only_normalization_replay_exact':True,'nested_reference_replay_exact':True};(s.ROOT/('model_verification_'+''.join(map(str,folds))+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='data':data()
 else:models(list(map(int,sys.argv[2].split(','))))
