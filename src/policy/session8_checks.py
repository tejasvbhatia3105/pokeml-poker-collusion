import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from session8_data import hand_data,targets,ROOT,C
from session8_event_model import load_models,score,add_features,FAMILIES
from session8_witness import hand_probability
def main():
    d=hand_data();report={'integrity':{},'replays':{}}
    for path in sorted(ROOT.glob('*_oof.parquet')):
        q=pl.read_parquet(path);assert len(q)==len(d) and q.select('pair_id','hand_id').n_unique()==len(q)
        assert q.join(d.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='anti').height==0
        assert np.isfinite(q.select('primary','secondary','score').to_numpy()).all()
        assert q['score'].min()>=0 and q['score'].max()<=1+1e-12
        mass=q.group_by('pair_id').agg(C('score').sum())['score'];assert mass.max()<=5+1e-10
        report['integrity'][path.stem]={'hands':len(q),'maximum_expected_selected':mass.max()}
    with threadpool_limits(limits=4):
        for mode in ['context','ledger','raw_events','shared_events','no_clock','exposure_cat','exposure_hist']:
            path=ROOT/f'{mode}_oof.parquet'
            if not path.exists():continue
            models,cols=load_models(mode);q=add_features(d,mode).join(pl.read_parquet(path).select('pair_id','hand_id',C('score').alias('expected')),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');error=0.
            for f in range(4):
                for b in FAMILIES:
                    z=q.filter((C('fold')==f)&(C('behavior_family')==b));p=score(models,cols,b,z,[f]);error=max(error,float(np.max(abs(p-z['expected'].to_numpy()))))
                    if f==0 and b==FAMILIES[0]:
                        r=z.reverse().with_columns(pl.lit(-999).alias('evidence'),pl.lit(-999).alias('evidence_rank'));a=score(models,cols,b,r,[f]);assert np.allclose(a[::-1],p,atol=1e-12)
            assert error<1e-10,(mode,error);report['replays'][mode]={'maximum_error':error,'row_and_unused_label_mutation':'passed'};print(mode,error,flush=True)
        mutated=d.with_columns(pl.when(C('fold')==0).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),pl.when(C('fold')==0).then(2).otherwise(C('subtype')).alias('subtype'),pl.when(C('fold')==0).then(99).otherwise(C('evidence_rank')).alias('evidence_rank'))
        for b in FAMILIES:
            before=targets(d,0,b);after=targets(mutated,0,b);assert all(np.array_equal(a,b) for a,b in zip(before,after))
    checked=0
    for n in range(1,7):
        p=np.linspace(.07,.79,n);q=hand_probability(p,np.zeros(n,dtype=int),1)[0];total=0.;post=np.zeros(n)
        for bits in itertools.product([0,1],repeat=n):
            bits=np.array(bits);weight=np.prod(np.where(bits,p,1-p))
            if bits.any():total+=weight;post+=bits*weight
        assert np.allclose(q,total,atol=1e-12) and np.allclose(post/total,p/q,atol=1e-12);checked+=2**n
    report['noisy_or_paths']=checked;report['held_out_target_mutation']='passed';(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
