import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
import session11_build_candidate as b
def main():
    old=b.conditioned;denoms=[]
    def track(p,k):
        p=np.asarray(p,dtype=np.float64);p=p/np.maximum(1,p.sum(1))[:,None];z=np.zeros(k+1);z[0]=1
        for s in p.sum(1):z=np.r_[z[0]*(1-s),z[1:k]*(1-s)+z[:k-1]*s,z[k]+z[k-1]*s]
        denoms.append(float(z[-1]));return old(p,k)
    b.conditioned=track;m=b.models('conditional_family');err=0.;rank_changes=0;cached=pl.read_parquet('artifacts/candidate_r30/evidence_scores.parquet')
    for path in sorted((b.ROOT/'eval_cache').glob('T*.parquet')):
        for (pid,),g in pl.read_parquet(path).group_by('pair_id'):
            g=g.sort('time','hand_id');s=np.mean([b.score(g,f,'conditional_family',m) for f in range(4)],axis=0);z=g.select('pair_id','hand_id').join(cached,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=max(err,float(np.max(abs(s-z['score'].to_numpy()))));rank_changes+=np.lexsort((g['hand_id'].to_numpy(),-s))[:5].tolist()!=np.lexsort((g['hand_id'].to_numpy(),-z['score'].to_numpy()))[:5].tolist()
    report={'conditional_distributions_checked':len(denoms),'minimum_condition_probability':min(denoms),'tiny_denominators_below_1e_8':int(sum(x<1e-8 for x in denoms)),'score_replay_max_error':err,'ranking_replay_mismatches':rank_changes};assert err<1e-12 and rank_changes==0
    (b.Path('artifacts/candidate_r30')/'inference_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
