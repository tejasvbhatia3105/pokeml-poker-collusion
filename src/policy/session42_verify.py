import json,itertools
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session42_precision import ROOT,BASE,OLD,PERMS,canonical,family_data,compute,replace,labels,target,rank
C=pl.col
def main():
 records=[]
 for family in ['directed_transfer','soft_play']:
  d,a,x,cols,ac,hc,ec=family_data(family);root=ROOT/family;cache=np.load(root/'equity.npz');values,info,states,eq=compute(d,a.reverse());np.testing.assert_array_equal(values,cache['values']);np.testing.assert_array_equal(states,cache['states']);np.testing.assert_array_equal(eq,cache['equity']);invariance=0
  for c in states[::max(1,len(states)//12)]:
   for s in PERMS:
    changed=[4*(int(v)//4)+s[int(v)%4] if v>=0 else -1 for v in c];assert canonical(changed)==tuple(c);assert canonical(changed[:2][::-1]+changed[2:])==tuple(c);invariance+=1
  er=0.;nr=0
  for i in np.flatnonzero((states>=0).sum(1)==7)[:20]:
   c=states[i];own=rank(c);deck=[v for v in range(52) if v not in c];wins=0
   for h in itertools.combinations(deck,2):
    other=rank(list(h)+list(c[2:]));wins+=int(own>other)+.5*int(own==other)
   expected=wins/990;er=max(er,abs(float(eq[i])-expected));nr+=1
  assert er<3e-8;x=replace(x,cols,values);g=a['row'].to_numpy();r=a['actor'].to_numpy();pp=np.zeros((len(d),2))
  for f in range(4):
   if family=='directed_transfer':y,tr,va=labels(d,a,f)
   else:y0,e,_=target(d,f);y=y0[g];tr=e[g];va=d['fold'].to_numpy()[g]==f
   assert not (tr&va).any();m=CatBoostClassifier();m.load_model(str(root/f'primary_fold{f}.cbm'));pp[g[va],r[va]]=m.predict_proba(x[va],thread_count=2)[:,1]
  if family=='directed_transfer':dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy()
  else:dw=np.ones_like(pp)
  q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');modelerr=float(abs((pp*dw).sum(1)-q['bg_primary'].to_numpy()).max());assert modelerr==0;records.append({'family':family,'models_replayed':4,'weighted_probability_error':modelerr,'raw_card_rebuild_reverse_order_exact':True,'suit_and_hole_order_invariance_cases':invariance,'independent_river_enumerations':nr,'river_float32_rounding_error':er})
 out=pl.read_parquet(ROOT/'event_oof.parquet').join(pl.read_parquet(BASE/'event_oof.parquet'),on=['pair_id','hand_id'],suffix='_base',validate='1:1');assert float((out['bg_secondary']-out['bg_secondary_base']).abs().max())==0;report={'families':records,'secondary_heads_unchanged':True};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
