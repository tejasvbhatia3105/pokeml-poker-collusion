import json
import numpy as np,polars as pl
from scipy.special import expit
from catboost import CatBoostClassifier,Pool
from session35_fold_likelihood import ROOT,OLD,data,labels,ordinary
def main():
 d,a,x,px,cfg,pc=data();g=a['row'].to_numpy();actor=a['actor'].to_numpy();dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();audit=json.load(open(ROOT/'audit.json'));records=[]
 for kind in ['normal_features','surprisal_offset']:
  pp=np.zeros((len(d),2));pool_error=0.
  for f in range(4):
   p=ordinary(px,f);surprise=-np.log(p[:,0].clip(1e-6,1));xx=np.column_stack([x,p,surprise]);y,tr,va=labels(d,a,f);item=next(r for r in audit if r['fold']==f and r['kind']==kind);assert abs(expit(surprise[tr]+item['intercept']).mean()-y[tr].mean())<1e-10;m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'primary_fold{f}.cbm'));off=surprise[va]+item['intercept'] if kind=='surprisal_offset' else np.zeros(va.sum());z=expit(m.predict(xx[va],prediction_type='RawFormulaVal',thread_count=2)+off);pool=m.predict_proba(Pool(xx[va],baseline=off),thread_count=2)[:,1];pool_error=max(pool_error,float(abs(pool-z).max()));pp[g[va],actor[va]]=z
  q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'conditional_primary.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(pp-q.select('actor0_primary','actor1_primary').to_numpy()).max());q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet').select('pair_id','hand_id','bg_primary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');weighted=float(abs((pp*dw).sum(1)-q['bg_primary'].to_numpy()).max());print('replay errors',kind,err,weighted,pool_error,flush=True);assert max(err,weighted)<1e-12 and pool_error<1e-7;records.append({'kind':kind,'models_replayed':4,'conditional_probability_error':err,'actor_weighted_error':weighted,'explicit_offset_vs_catboost_pool_error':pool_error})
 report={'feature_join_matches_existing_action_states':True,'models':records,'target_audit':audit};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
