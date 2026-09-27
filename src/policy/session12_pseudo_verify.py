import os,json,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import polars as pl,numpy as np
ROOT=Path('artifacts/evidence_session12/unlabelled_pseudo');C=pl.col

def main():
 reg=pl.read_csv('data/evaluation_pairs.csv');allowed={x for x in reg['pair_id'] if hashlib.sha256(x.encode()).digest()[0]%2==0};known=set(pl.read_csv('data/development_labels.csv')['pair_id']);assert not allowed&known;folds=json.load(open('artifacts/policy/table_folds.json'));cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];assert not set(cols)&{'pair_id','hand_id','fold','evidence','evidence_rank','subtype'};report=[]
 for f in range(4):
  files=list((ROOT/f'outer{f}').glob('T*.parquet'));expected={t for t,v in folds.items() if v!=f};assert {p.stem for p in files}==expected;q=pl.read_parquet(files);assert q.select('pair_id','hand_id').n_unique()==len(q);assert set(q['pair_id'])<=allowed;assert (q['pool_fold']!=f).all() and (q['teacher_fold']==f).all();assert np.isfinite(q.select(cols).to_numpy()).all();assert q.select(C('eligible_primary')|C('eligible_secondary')).to_series().all()
  for name in ['primary','secondary']:
   x=q.filter(C('eligible_'+name))['pseudo_'+name];assert ((x<.02)|(x>.9)).all()
  counts=q.group_by('pair_id').agg(((C('eligible_primary')&(C('pseudo_primary')>.9))|(C('eligible_secondary')&(C('pseudo_secondary')>.9))).sum().alias('positive_hands'));assert counts['positive_hands'].min()>=2
  gate=pl.read_parquet(f'artifacts/evidence_session12/pseudo_pair_gate/gate_fold{f}.parquet');assert set(gate['pair_id'])==set(q['pair_id']);assert (gate['pool_fold']!=f).all();expected_keep=(gate['pair_risk']>=.9)&(gate['pair_family']==gate['behavior_family']);assert (expected_keep==gate['keep']).all();kept=q.join(gate.filter(C('keep')).select('pair_id'),on='pair_id',validate='m:1');report.append({'outer_fold':f,'scanned_tables':len(files),'pseudo_pairs':q['pair_id'].n_unique(),'pseudo_hands':len(q),'gated_pairs':kept['pair_id'].n_unique(),'gated_hands':len(kept),'minimum_confident_event_hands':int(counts['positive_hands'].min()),'teacher_training_excludes_outer_fold':True})
 result={'hash_sample_registry_count':len(allowed),'registry_excludes_all_public_labelled_pairs':True,'folds':report,'event_teacher':'archived primary/fallback Cat and HGB heads indexed by held-out outer fold; raw gameplay inputs contain no model scores/labels','pair_teacher':'fresh 600-tree confirmed-label-only models, no outer early stopping or global pseudo ancestry','limitations':['Pseudo events remain model hypotheses, never certified planted labels.','Original ordinary-policy feature caches and their evaluation averaging are retained.','Public validation folds have been reused throughout research.']};(ROOT/'provenance_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
