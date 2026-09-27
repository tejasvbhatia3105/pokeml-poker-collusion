import os,json,itertools,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from evidence_data import load
from session6_priority import inclusion,training_targets
from session6_priority_model import load_models,score,ROOT
def brute(a,b):
    scale=np.maximum(1,a+b);p=np.stack([1-(a+b)/scale,a/scale,b/scale],1);out=np.zeros(len(a))
    for state in itertools.product(range(3),repeat=len(a)):
        probability=np.prod(p[np.arange(len(a)),state])
        chosen=([i for i,k in enumerate(state) if k==1]+[i for i,k in enumerate(state) if k==2])[:5]
        out[chosen]+=probability
    return out
def main():
    prefix=os.environ.get('PRIORITY_PREFIX','priority')
    rng=np.random.default_rng(611);errors=[]
    for n in [1,4,5,6,7]:
        for trial in range(5):
            a=rng.random(n);b=rng.random(n)
            errors.append(float(np.max(np.abs(inclusion(a,b)-brute(a,b)))))
            assert inclusion(a,b).sum()<=5+1e-10
    for a,b in [(np.ones(8),np.zeros(8)),(np.zeros(8),np.ones(8)),(np.zeros(8),np.zeros(8))]:
        assert np.allclose(inclusion(a,b),brute(a,b),atol=1e-12)
    assert max(errors)<1e-12
                                                                            
                                                                              
    y=rng.integers(0,2,60);sub=rng.integers(0,3,60);tp=rng.random(60);tr=np.arange(60)<40;t=np.arange(60);groups=np.arange(60).reshape(6,10);rank=rng.permutation(60)
    for ordered in (False,True):
        first=training_targets(y,sub,tp,tr,t,groups,rank if ordered else None)
        y2=y.copy();y2[~tr]=1-y2[~tr];s2=sub.copy();s2[~tr]=rng.integers(0,3,(~tr).sum());r2=rank.copy();r2[~tr]=rng.permutation(r2[~tr])
        second=training_targets(y2,s2,tp,tr,t,groups,r2 if ordered else None)
        assert all(np.array_equal(a,b) for a,b in zip(first,second))
        assert all(not a[~tr].any() for a in first)
    d,_=load();p=Path('artifacts/policy')
    extra=pl.read_parquet(list((p/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((p/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1')
    d=d.join(extra,on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'],validate='1:1')
    models,cols=load_models(prefix);saved=pl.read_parquet(ROOT/f'{prefix}_oof.parquet');replay=[];shuffle=[];label=[]
    for f in range(4):
        for b in models:
            q=d.filter((pl.col('fold')==f)&(pl.col('behavior_family')==b));pred=score(models,cols,b,q,[f])
            expected=q.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['score'].to_numpy();replay.append(float(np.max(np.abs(pred-expected))))
            shuffled=q.sample(fraction=1,shuffle=True,seed=17);other=score(models,cols,b,shuffled,[f])
            back=q.select('pair_id','hand_id').join(shuffled.select('pair_id','hand_id').with_columns(pl.Series('prediction',other)),on=['pair_id','hand_id'],maintain_order='left')['prediction'].to_numpy();shuffle.append(float(np.max(np.abs(pred-back))))
            mutated=q.with_columns((1-pl.col('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'));label.append(float(np.max(np.abs(pred-score(models,cols,b,mutated,[f])))))
    assert max(replay)<1e-12 and max(shuffle)<1e-12 and max(label)<1e-12
    sha=hashlib.sha256(Path('artifacts/candidate_r27/submission.csv').read_bytes()).hexdigest();assert sha=='1cef3d932b0fea2e669980ab8b4f367f89352b5aaf1553bb3a86b2546873f9ae'
    result={'enumeration_cases':28,'enumeration_max_error':max(errors),'heldout_target_mutation_passed':True,'saved_model_replay_hands':len(d),'replay_max_error':max(replay),'row_shuffle_max_error':max(shuffle),'inference_label_mutation_max_error':max(label),'r27_sha256':sha}
    (ROOT/f'{prefix}_checks.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
