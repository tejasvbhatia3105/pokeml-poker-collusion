import json,csv,hashlib
from pathlib import Path
import numpy as np,polars as pl
import session46_paired_inference as infer
OUT=Path('artifacts/candidate_r31');C=pl.col
def main(out=OUT):
 OUT=out
 m=infer.load_models();ordinary=infer.conditioned;denoms=[]
 def track(p,k):
  p=np.asarray(p,dtype=np.float64);p=p/np.maximum(1,p.sum(1))[:,None];z=np.zeros(k+1);z[0]=1
  for s in p.sum(1):z=np.r_[z[0]*(1-s),z[1:k]*(1-s)+z[:k-1]*s,z[k]+z[k-1]*s]
  denoms.append(float(z[-1]));return ordinary(p,k)
 infer.conditioned=track;err=0.;rankerr=0;hands=0
 with (OUT/'submission.csv').open(newline='') as f:rows={r['pair_id']:r for r in csv.DictReader(f)}
 for path in sorted((OUT/'inference').glob('T*.parquet')):
  pred=pl.read_parquet(path);pcols=[c for c in pred.columns if c.startswith('new_') or c.startswith('score')];d=pl.read_parquet(Path('artifacts/evidence_session11/eval_cache')/path.name).join(pred.select('pair_id','hand_id',*pcols),on=['pair_id','hand_id'],validate='1:1');assert np.isfinite(pred.select(pcols).to_numpy()).all()
  for (pid,),g in d.group_by('pair_id'):
   g=g.sort('time','hand_id');ss=[infer.score_pair(g,f,m) for f in range(4)];score=np.mean(ss,0);err=max(err,float(abs(score-g['score'].to_numpy()).max()),max(float(abs(v-g[f'score_{f}'].to_numpy()).max()) for f,v in enumerate(ss)));chosen=g.with_columns(pl.Series('replay',score)).sort('replay','hand_id',descending=[True,False])['hand_id'].to_list()[:5];rankerr+=chosen!=[rows[pid][f'evidence_hand_{i}'] for i in range(1,6)];hands+=len(g)
 assert err<1e-12 and rankerr==0;report={'hands_replayed':hands,'conditional_distributions':len(denoms),'minimum_condition_probability':min(denoms),'denominators_below_1e_minus_8':sum(v<1e-8 for v in denoms),'score_replay_max_error':err,'csv_ranking_mismatches':rankerr,'sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest(),'scope':'all posterior selection scores replayed from saved event probabilities; raw-model generalization audited separately'};(OUT/'inference_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
