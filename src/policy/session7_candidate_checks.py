import os,csv,json,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
def main():
    out=Path('artifacts/candidate_r29');d=pl.read_parquet(out/'evidence_scores.parquet');prior=pl.read_parquet('artifacts/candidate_r28/evidence_scores.parquet');key=['pair_id','hand_id'];oldcols=['r27_base_score','priority_score']
    z=d.join(prior.select(*key,*oldcols),on=key,suffix='_old',validate='1:1');assert len(z)==len(d)==len(prior);errors={c:float((z[c]-z[c+'_old']).abs().max()) for c in oldcols};assert max(errors.values())<1e-12,errors
    def read(path):
        with path.open(newline='') as f:return list(csv.DictReader(f))
    old=read(Path('artifacts/candidate_r28/submission.csv'));new=read(out/'submission.csv');oldmap={r['pair_id']:r for r in old};newmap={r['pair_id']:r for r in new};ecols=[f'evidence_hand_{i}' for i in range(1,6)];changed=0;replay=0
    assert len(old)==len(new)==112540
    for a,b in zip(old,new):assert all(a[k]==b[k] for k in a if k not in ecols);changed+=any(a[k]!=b[k] for k in ecols)
    for (pid,),g in d.group_by('pair_id'):
        formula=.25*g['r27_base_score'].to_numpy()+.25*g['priority_score'].to_numpy()+.5*g['hist_eventblend_score'].to_numpy();assert np.allclose(g['base_score'].to_numpy(),formula,atol=1e-15)
        hands=g.sort(['base_score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5];assert hands==[newmap[pid][k] for k in ecols][:len(hands)]
        legacy=g.with_columns(((pl.col('r27_base_score')+pl.col('priority_score'))*.5).alias('old_score')).sort(['old_score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5];assert legacy==[oldmap[pid][k] for k in ecols][:len(legacy)]
        assert (g['time'].to_numpy()>=.6).all() and g['hist_eventblend_score'].sum()<=5+1e-10;replay+=1
    sha=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest();report={'rows':len(new),'changed_evidence_rows':changed,'rescored_pairs':replay,'r28_prediction_max_error':errors,'r28_and_r29_rankings_replayed':True,'non_evidence_strings_unchanged':True,'evaluation_phase_only':True,'candidate_hand_scores':len(d),'sha256':sha};(out/'lineage_checks.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
