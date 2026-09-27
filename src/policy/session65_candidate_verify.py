import json,csv,hashlib
from pathlib import Path
import numpy as np,polars as pl
import session65_list_pressure_inference as infer
import session46_paired_inference as pressure
OUT=Path('artifacts/candidate_r33');C=pl.col
EXPECTED={30:'d2e6e55063d8be986d9d009c036db4c64c55ffb23a515db468871be80a92195f',31:'14fcc29b4e81e186c12552ddfc2f98f6bdf6812a26cc315ca394943885ff97f9',32:'e6e3303e55feccdfb44e02dff5b83f11521f441103beadddf167caad0d416038'}
def main():
 models=infer.load_models();ordinary=infer.conditioned;denoms={'pressure':[],'tree':[]}
 def tracker(kind):
  def track(p,k):
   a=np.asarray(p,dtype=np.float64);a=a/np.maximum(1,a.sum(1))[:,None];z=np.zeros(k+1);z[0]=1
   for s in a.sum(1):z=np.r_[z[0]*(1-s),z[1:k]*(1-s)+z[:k-1]*s,z[k]+z[k-1]*s]
   denoms[kind].append(float(z[-1]));return ordinary(p,k)
  return track
 pressure.conditioned=tracker('pressure');infer.conditioned=tracker('tree')
 with (OUT/'submission.csv').open(newline='') as f:reader=csv.DictReader(f);fieldnames=reader.fieldnames;rows=list(reader)
 by={r['pair_id']:r for r in rows};error=0;rankerror=0;hands=pairs=0
 for path in sorted((OUT/'inference').glob('T*.parquet')):
  pred=pl.read_parquet(path);cols=[c for c in pred.columns if c.startswith('new_') or c.startswith('score')]+infer.EXTRA_COLUMNS;assert len(set(cols))==len(cols);d=pl.read_parquet(Path('artifacts/evidence_session11/eval_cache')/path.name).join(pred.select('pair_id','hand_id',*cols),on=['pair_id','hand_id'],validate='1:1');assert np.isfinite(pred.select(cols).to_numpy()).all()
  for (pid,),g in d.group_by('pair_id'):
   g=g.sort('time','hand_id');scores=[infer.score_pair(g,f,models) for f in range(4)];score=np.mean(scores,0);error=max(error,float(abs(score-g['score'].to_numpy()).max()),*[float(abs(s-g[f'score_{f}'].to_numpy()).max()) for f,s in enumerate(scores)]);chosen=g.with_columns(pl.Series('replay',score)).sort('replay','hand_id',descending=[True,False])['hand_id'].to_list()[:5];rankerror+=chosen!=[by[pid][f'evidence_hand_{i}'] for i in range(1,6)];hands+=len(g);pairs+=1
 assert error==0 and rankerror==0
 for r,sha in EXPECTED.items():assert hashlib.sha256(Path(f'artifacts/candidate_r{r}/submission.csv').read_bytes()).hexdigest()==sha
 with Path('artifacts/candidate_r32/submission.csv').open(newline='') as f:source=list(csv.DictReader(f))
 ecols=[f'evidence_hand_{i}' for i in range(1,6)];changed=0;families={}
 for old,new in zip(source,rows,strict=True):
  assert all(old[c]==new[c] for c in fieldnames if c not in ecols)
  if float(old['risk_score'])<.05:assert old==new
  if old!=new:
   assert float(old['risk_score'])>=.05 and old['predicted_behavior'] in ['directed_transfer','soft_play','coordinated_isolation'];changed+=1;fam=old['predicted_behavior'];families[fam]=families.get(fam,0)+1
 report={'hands_replayed':hands,'pairs_replayed':pairs,'score_replay_error':error,'csv_ranking_mismatches':rankerror,'conditional_distributions':{k:len(v) for k,v in denoms.items()},'minimum_condition_probability':{k:min(v) for k,v in denoms.items()},'denominators_below_1e_minus_8':{k:sum(x<1e-8 for x in v) for k,v in denoms.items()},'sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest()};(OUT/'inference_replay.json').write_text(json.dumps(report,indent=2));pres={'compared_to':'R32 .92907 user-confirmed','source_hash':EXPECTED[32],'original_R30_R31_R32_hashes_preserved':True,'all_risk_behavior_and_non_evidence_strings_unchanged':True,'all_below005_rows_preserved':True,'changed_evidence_rows_vs_R32':changed,'changed_by_predicted_family':families};(OUT/'source_preservation.json').write_text(json.dumps(pres,indent=2));print(json.dumps({'replay':report,'preservation':pres},indent=2))
if __name__=='__main__':main()
