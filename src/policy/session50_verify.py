import json,itertools
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from cards import rank
from session50_matchup import ROOT,load,compute,evaluate,canonical,OLD
def aggregate(d,a,x,root,iso):
 fv=d['fold'].to_numpy();g=a['row'].to_numpy();r=a['actor'].to_numpy();pp=np.zeros((len(d),2))
 for f in range(4):
  va=(fv==f) if iso else (fv[g]==f)
  for k in range(2 if iso else 1):
   path=root/f'event{k+1}_fold{f}.cbm'
   if not path.exists():path=root/f'primary_fold{f}.cbm'
   m=CatBoostClassifier();m.load_model(str(path));p=m.predict_proba(x[va],thread_count=2)[:,1]
   if iso:pp[va,k]=p
   else:pp[g[va],r[va]]=p
 if iso:return pp
 dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if d['behavior_family'][0]=='directed_transfer' else np.ones_like(pp)
 return (pp*dw).sum(1)[:,None]
def main():
 records=[];allstates=[];baseline=pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/event_oof.parquet');newref=pl.read_parquet(ROOT/'event_oof.parquet');folders=['artifacts/evidence_session37_bet_fold/paired_hand','artifacts/evidence_session38_soft_bet_fold/paired_soft','artifacts/evidence_session41_isolation_bet_fold/paired_fold']
 from pathlib import Path
 for family,folder in zip(['directed_transfer','soft_play','coordinated_isolation'],folders):
  d,a,x,cols=load(family);saved=np.load(ROOT/family/'features.npz');states,ex=compute(d.reverse(),a.reverse());np.testing.assert_array_equal(states,saved['states']);np.testing.assert_array_equal(ex,saved['x']);allstates.append(states);iso=family=='coordinated_isolation';additional=np.full((len(d),7),-2.) if iso else ex
  if iso:additional[a['row'].to_numpy()]=ex
  errors=[]
  for xx,root,ref in [(x,Path(folder),baseline),(np.column_stack([x,additional]),ROOT/family,newref)]:
   p=aggregate(d,a,xx,root,iso);q=d.select('pair_id','hand_id').join(ref,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(p-q.select(['bg_primary','bg_secondary'] if iso else ['bg_primary']).to_numpy()).max());assert err==0;errors.append(err)
  records.append({'family':family,'new_models_replayed':8 if iso else 4,'original_models_replayed':8 if iso else 4,'baseline_and_new_probability_errors':errors,'reverse_raw_feature_rebuild_exact':True})
 states=np.concatenate(allstates);rng=np.random.default_rng(5001);pick=states[rng.choice(len(states),100,replace=False)];canonical_count=0
 for cs in pick:
  for perm in itertools.permutations(range(4)):
   mapped=[4*(int(c)//4)+perm[int(c)%4] if c>=0 else -1 for c in cs];mapped[:2]=mapped[:2][::-1];mapped[2:4]=mapped[2:4][::-1];assert canonical(mapped)==tuple(cs);canonical_count+=1
 post=states[states[:,4]>=0];post=post[rng.choice(len(post),100,replace=False)];out=evaluate(post);swap=post.copy();swap[:,:2]=post[:,2:4];swap[:,2:4]=post[:,:2];sw=evaluate(swap);assert np.allclose(out[:,0]+sw[:,0],1,atol=1e-7);assert np.array_equal(out[:,2],sw[:,2]);assert np.array_equal(out[:,3],-sw[:,3]);np.testing.assert_array_equal(out[:,4],sw[:,5]);exact_error=0;cases=0
 for nb in [3,4,5]:
  group=states[(states[:,4:]>=0).sum(1)==nb][:8]
  for cs in group:
   used=set(int(v) for v in cs if v>=0);deck=[v for v in range(52) if v not in used];w=t=n=0
   for tail in itertools.combinations(deck,5-nb):
    board=list(cs[4:4+nb])+list(tail);a=rank(list(cs[:2])+board);b=rank(list(cs[2:4])+board);w+=a>b;t+=a==b;n+=1
   py=np.array([(w+.5*t)/n,w/n,t/n]);c=evaluate(cs[None])[0,:3];exact_error=max(exact_error,float(abs(py-c).max()));cases+=1
 assert exact_error<3e-8;report={'families':records,'suit_and_hole_order_invariance_cases':canonical_count,'postflop_actor_swap_cases':100,'independent_python_runout_enumerations':cases,'python_enumeration_probability_error':exact_error,'scope':'Pair-private-information equity; not EV; future board and outsiders holdings excluded','preflop_sampling':2048};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
