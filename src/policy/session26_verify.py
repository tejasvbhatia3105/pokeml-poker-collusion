import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session26_exact_fold_witness import actions
C=pl.col;ROOT=Path('artifacts/evidence_session26_exact_fold');OLD=Path('artifacts/evidence_session25_persistent_actor')
def main():
 d=hand_data().filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');a,cols=actions(d);saved=pl.read_parquet(ROOT/'fold_actions.parquet');keys=['pair_id','hand_id','actor'];assert a.sort(keys).equals(saved.sort(keys));a=saved;cfg=json.load(open(ROOT/'columns.json'));assert cols==cfg['action'];ax=a.select(cols).to_numpy();hx=d.select(cfg['hand']).to_numpy();g=a['row'].to_numpy();r=a['actor'].to_numpy();fv=d['fold'].to_numpy();dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();errors={}
 for kind in ['action','action_hand']:
  pp=np.zeros((len(d),2));x=ax if kind=='action' else np.column_stack([ax,hx[g]])
  for f in range(4):
   m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'primary_fold{f}.cbm'));va=fv[g]==f;pp[g[va],r[va]]=m.predict_proba(x[va],thread_count=2)[:,1]
  q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'conditional_primary.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(pp-q.select('actor0_primary','actor1_primary').to_numpy()).max());out=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet').select('pair_id','hand_id','bg_primary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');weighted=float(abs((pp*dw).sum(1)-out['bg_primary'].to_numpy()).max());assert max(err,weighted)<1e-12;errors[kind]={'models_replayed':4,'conditional_error':err,'actor_weighted_error':weighted}
 report={'action_feature_rebuild_exact':True,'one_fold_action_per_pair_hand_actor':True,'model_replays':errors,'target_audit':json.load(open(ROOT/'audit.json')),'limitation':'fold-tier and persistent-actor hypotheses are inferred public training labels; they are not certified action-level truth'};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
