import os,csv,json,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from scipy.special import logit
from session4_evidence_model import FAMILIES,ROOT
def unary(g,weights):
                                                                        
                                                                            
    g=g.with_columns((((pl.col('time')-.6)/.4)*.6).alias('time')).sort('time','hand_id');n=len(g);scores=g['r27_base_score'].to_numpy();idx=np.lexsort((g['hand_id'].to_numpy(),-scores))[:12]
    if n<12:return g[idx]['hand_id'].to_list()[:5]
    t=g[idx]['time'].to_numpy()*5000;sc=scores[idx];tc=np.argsort(np.argsort(t));order=np.argsort(t);gaps=np.diff(t[order])/3000;left=np.ones(12);right=np.ones(12);left[order[1:]]=gaps;right[order[:-1]]=gaps;mass=np.cumsum(scores)
    U=np.stack([logit(np.clip(sc,1e-5,1-1e-5)),t/3000,idx/n,tc/11,np.log1p(mass[idx]-sc),np.log1p(mass[-1]-mass[idx]),left,right],1).astype('float32');f=FAMILIES.index(g['behavior_family'][0]);coef=weights[:8]+weights[8+f*8:16+f*8];hand=g[idx]['hand_id'].to_numpy();return hand[np.lexsort((hand,-U@coef))[:5]].tolist()
def main():
    out=Path('artifacts/candidate_r28');d=pl.read_parquet(out/'evidence_scores.parquet');weights=np.mean([np.load(ROOT/f'nested_blend/unary_weights_fold{f}.npy') for f in range(4)],0)
    def read(path):
        with path.open(newline='') as f:return list(csv.DictReader(f))
    old=read(Path('artifacts/candidate_r27/submission.csv'));new=read(out/'submission.csv');oldmap={r['pair_id']:r for r in old};newmap={r['pair_id']:r for r in new};ecols=[f'evidence_hand_{i}' for i in range(1,6)];r27_mismatches=[];total=0
    assert len(old)==len(new)==112540
    for a,b in zip(old,new):assert all(a[k]==b[k] for k in a if k not in ecols)
    for (pid,),g in d.group_by('pair_id'):
        chosen=g.sort(['base_score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5];assert chosen==[newmap[pid][k] for k in ecols][:len(chosen)]
        legacy=unary(g,weights)
        if legacy!=[oldmap[pid][k] for k in ecols][:len(legacy)]:r27_mismatches.append(pid)
        assert np.allclose(g['base_score'].to_numpy(),.5*g['r27_base_score'].to_numpy()+.5*g['priority_score'].to_numpy(),atol=1e-15);assert (g['time'].to_numpy()>=.6).all();assert g['priority_score'].sum()<=5+1e-10;total+=1
    assert not r27_mismatches, r27_mismatches[:10]
    report={'pairs_rankings_replayed':total,'r27_rankings_reproduced':total,'r27_rank_mismatches':r27_mismatches,'non_evidence_strings_unchanged':True,'candidate_hand_scores':len(d),'evaluation_phase_only':True,'inclusion_sums_at_most_five':True,'sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()};(out/'lineage_checks.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
