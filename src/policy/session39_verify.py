import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session39_bet_call import ROOT,BASE,OLD,PREV,data,extra_features,targets,noisy_or
C=pl.col
def main():
 d,a,cfg=data();extra,align=extra_features(d,a.reverse());assert extra.equals(pl.read_parquet(ROOT/'action_features.parquet'));assert align.equals(pl.read_parquet(ROOT/'alignment.parquet'));assert (align['bet_action_no']<align['call_action_no']).all();config=json.load(open(ROOT/'config.json'));g=a['row'].to_numpy();r=a['actor'].to_numpy();fv=d['fold'].to_numpy();bag=2*g+r;x=np.column_stack([a.select(cfg['action']).to_numpy(),d.select(cfg['hand']).to_numpy()[g],extra.select(config['paired_columns']).to_numpy()]);pp=np.zeros((len(d),2));audit=[]
 for f in range(4):
  y,e,donor=targets(d,f);tr=e[g]&(r==donor[g]);va=fv[g]==f;assert not (tr&va).any();m=CatBoostClassifier();m.load_model(str(ROOT/f'secondary_fold{f}_em2.cbm'));p=m.predict_proba(x[va],thread_count=2)[:,1];pp[fv==f]=noisy_or(p,bag[va],2*len(d)).reshape(-1,2)[fv==f];audit.append({'fold':f,'training_actions':int(tr.sum()),'heldout_actions':int(va.sum()),'overlap':0})
 q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'conditional_secondary.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(pp-q.select('actor0_secondary','actor1_secondary').to_numpy()).max());assert err==0;dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/'paired_call/event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');weighted=float(abs((pp*dw).sum(1)-q['bg_secondary'].to_numpy()).max());assert weighted==0
 for kind in ['original_call','paired_call']:
  out=pl.read_parquet(ROOT/kind/'event_oof.parquet').join(pl.read_parquet(BASE/'event_oof.parquet'),on=['pair_id','hand_id'],suffix='_base',validate='1:1');assert float((out['bg_primary']-out['bg_primary_base']).abs().max())==0
 report={'raw_bet_call_features_rebuild_reverse_order_exact':True,'aligned_calls':len(a),'final_models_replayed':4,'conditional_probability_error':err,'actor_weighted_error':weighted,'primary_unchanged':True,'targets':audit,'intermediate_EM_models':'saved, final inference uses EM2 only'};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
