import os,json,math
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session92_chip_signatures as s
C=pl.col
def model(path,x):
 m=CatBoostClassifier();m.load_model(str(path));p=m.predict_proba(x,thread_count=2)[:,1];np.testing.assert_array_equal(p,m.predict_proba(x[::-1],thread_count=2)[:,1][::-1]);return p
def mutate(d,f):
 return d.with_columns(*[pl.when(C('fold')==f).then(pl.lit(v)).otherwise(C(k)).alias(k) for k,v in [('evidence',0),('evidence_rank',-999),('subtype',-999)]])
def main():
 states=s.state();full=s.hand_data();saved=pl.read_parquet(s.ROOT/'event_oof.parquet');base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');count=0;actions=0;scalar=0;exclusions=0
 for fam in ['directed_transfer','soft_play','coordinated_isolation']:
  raw=pl.read_parquet(s.ROOT/f'{fam}_raw.parquet');keys=['action_row','pair_id','hand_id','table_id','action_no']+(['partner'] if fam!='coordinated_isolation' else []);fresh=s.raw_queries(raw.select(keys));np.testing.assert_array_equal(raw.select(s.RAW).to_numpy(),fresh.select(s.RAW).to_numpy());x,cols=s.formulas(fresh)
                                                                                     
  for r in fresh.sample(n=min(128,len(fresh)),seed=9201).iter_rows(named=True):
   A,T,P,S,K,B=[int(r[c]) for c in s.RAW];v=[]
   for a,b in [(A,P),(A-K,P+K),(T,P),(A,S),(A,B)]:
    for n,d in s.RATIOS:
     error=(a-(n*b)/d)/B;v.extend([math.copysign(math.log1p(abs(error)),error),a*d==n*b,a==(n*b)//d])
   v.append(math.log1p((T-A)/B));np.testing.assert_allclose(x[r['action_row']],np.array(v,dtype=np.float32),atol=2e-6,rtol=2e-6);scalar+=1
  lc=json.load(open(s.ROOT/'columns.json'))['ledger'];ledger=pl.read_parquet('artifacts/evidence_session8/ledger_actions.parquet').with_columns(C('action_no').cast(pl.Int64));ll=raw.select('hand_id','action_no').join(ledger,on=['hand_id','action_no'],validate='m:1',maintain_order='left').select(lc).to_numpy();extra=np.column_stack([x,ll]).astype(np.float32);np.testing.assert_array_equal(extra,np.load(s.ROOT/f'{fam}_features.npz')['x']);actions+=len(raw)
  if fam=='coordinated_isolation':
   _,d,a,ac=s.pressure_data();g=a['row'].to_numpy();fv=d['fold'].to_numpy();hc=json.load(open('artifacts/evidence_session59_pressure_equity/config.json'))['hand_columns'];xx=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x'],extra]);pred=np.zeros((len(d),2));fm=full['behavior_family'].to_numpy()==fam
   for f in range(4):
    original=s.targets(full,f,fam);changed=s.targets(mutate(full,f),f,fam)
    for z,zz in zip(original[:4],changed[:4]):np.testing.assert_array_equal(z,zz)
    va=fv[g]==f
    for head in range(2):
     tr=original[2+head][fm][g];assert not (tr&va).any();exclusions+=1
     for em in range(3):
      p=model(s.ROOT/f'isolation_head{head+1}_fold{f}_em{em}.cbm',xx[va]);assert np.isfinite(p).all();count+=1
     hp=s.noisy_or(p,g[va],len(d));pred[fv==f,head]=hp[fv==f]
   expect=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left').select('bg_primary','bg_secondary').to_numpy();np.testing.assert_array_equal(pred,expect)
  else:
   v=states[fam];d,a=v['d'],v['a'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];xx=np.column_stack([v['x'],extra]);dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2));pp=np.zeros((len(d),2))
   for f in range(4):
    if fam=='directed_transfer':
     y,tr,va=s.labels(d,a,f);yy,tt,vv=s.labels(mutate(d,f),a,f)
    else:
     y,e,_=s.target(d,f);yy,ee,_=s.target(mutate(d,f),f);tr=e[g];tt=ee[g];va=fv[g]==f;vv=va
    for z,zz in [(y,yy),(tr,tt),(va,vv)]:np.testing.assert_array_equal(z,zz)
    assert not (tr&va).any();exclusions+=1;pp[g[va],actor[va]]=model(s.ROOT/f'{fam}_primary_fold{f}.cbm',xx[va]);count+=1
   expect=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['bg_primary'].to_numpy();np.testing.assert_array_equal((pp*dw).sum(1),expect)
   z=saved.join(d.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi').join(base,on=['pair_id','hand_id'],suffix='_base',validate='1:1');np.testing.assert_array_equal(z['bg_secondary'].to_numpy(),z['bg_secondary_base'].to_numpy())
 report={'saved_models_replayed':count,'raw_actions_verified':actions,'scalar_formula_rows_checked':scalar,'target_exclusion_checks':exclusions,'final_event_score_error':0,'feature_cache_error':0,'row_permutation_error':0,'direct_soft_secondary_unchanged':True};(s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
