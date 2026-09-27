import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session26_exact_fold_witness import actions
from session29_shared_fold_witness import target
C=pl.col;ROOT=Path('artifacts/evidence_session29_shared_folds')
def main():
 d=hand_data().filter(C('behavior_family').is_in(['directed_transfer','soft_play'])).drop('row').with_row_index('row');a,ac=actions(d);saved=pl.read_parquet(ROOT/'fold_actions.parquet');keys=['pair_id','hand_id','actor'];assert a.sort(keys).equals(saved.sort(keys));a=saved;cfg=json.load(open(ROOT/'columns.json'));g=a['row'].to_numpy();r=a['actor'].to_numpy();fv=d['fold'].to_numpy();isdir=d['behavior_family'].to_numpy()=='directed_transfer';x=np.column_stack([a.select(ac).to_numpy(),d.select(cfg['hand']).to_numpy()[g]]);dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',how='left',validate='m:1',maintain_order='left').select(C('actor0').fill_null(1),C('actor1').fill_null(1)).to_numpy();errors={}
 for f in range(4):
  original=target(d,f);mut=d.with_columns(pl.when(C('fold')==f).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when(C('fold')==f).then(17).otherwise(C('evidence_rank')).alias('evidence_rank'),pl.when(C('fold')==f).then(-C('time')).otherwise(C('time')).alias('time'));changed=target(mut,f)
  for a0,b0 in zip(original,changed):assert np.array_equal(a0,b0)
 for kind in ['soft_only','pooled']:
  pp=np.zeros((len(d),2));xx=x if kind=='soft_only' else np.column_stack([x,isdir[g]])
  for f in range(4):
   m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'primary_fold{f}.cbm'));va=fv[g]==f
   if kind=='soft_only':va&=~isdir[g]
   pp[g[va],r[va]]=m.predict_proba(xx[va],thread_count=2)[:,1]
  q=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'conditional_primary.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=float(abs(pp-q.select('actor0_primary','actor1_primary').to_numpy()).max());out=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet').select('pair_id','hand_id','bg_primary'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');use=~isdir if kind=='soft_only' else np.ones(len(d),bool);weighted=float(abs((pp*dw).sum(1)[use]-out['bg_primary'].to_numpy()[use]).max());assert max(err,weighted)<1e-12;errors[kind]={'models':4,'conditional_error':err,'hand_probability_error':weighted}
 report={'raw_fold_feature_rebuild_exact':True,'validation_label_rank_time_mutation_changes_training_targets':False,'model_replays':errors,'limitation':'hypotheses chosen on reused public validation; mutation checks pipeline isolation, not pristine model selection'};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
