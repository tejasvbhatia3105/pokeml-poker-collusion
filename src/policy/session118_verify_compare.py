import os,json,hashlib
import numpy as np
import polars as pl
import torch
import session118_grounded_context as s
from session12_compare import compare
C=pl.col

def verify():
    device=os.environ.get('LIST_DEVICE','mps');d,cols=s.prepare();records=[];parts=[]
    for f in range(4):
        groups,raw,train,valid,mu,sd,mins=s.pack(d,cols,f);ts=tuple(x.to(device) for x in raw)
                                                                                  
                                                       
        mutated=d.with_columns(pl.when(C('fold')==f).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'),
            pl.when(C('fold')==f).then(None).otherwise(C('evidence_rank')).alias('evidence_rank'))
        _,mr,mt,mv,mmu,msd,mmins=s.pack(mutated,cols,f)
        assert np.array_equal(train,mt) and np.array_equal(valid,mv) and np.array_equal(mu,mmu) and np.array_equal(sd,msd) and mins==mmins
        for old,new in zip(raw,mr):assert torch.equal(old[train],new[train])
        predictions={}
        for kind in s.CONFIG['kinds']:
            predictions[kind]=[]
            for seed in s.CONFIG['seeds']:
                path=s.ROOT/f'{kind}_fold{f}_seed{seed}.pt';ck=torch.load(path,map_location=device,weights_only=False)
                assert np.array_equal(ck['mu'],mu) and np.array_equal(ck['sd'],sd) and ck['minimums']==mins
                assert ck['train_pair_ids']==[groups[i]['pair_id'][0] for i in train]
                assert not set(ck['train_pair_ids'])&set(ck['valid_pair_ids'])
                m=s.base.Model(len(mu),kind).to(device);m.load_state_dict(ck['state_dict']);m.eval()
                pred=s.predict(m,ts,valid);predictions[kind].append(pred)
                saved=np.load(s.ROOT/f'{kind}_fold{f}_seed{seed}_pred.npz')['probability']
                error=float(abs(saved-np.concatenate(pred)).max());assert error<1e-7,error
                                                                                
                X,P,M,T,V,D,K=s.batches(ts,valid[:3])
                with torch.no_grad():
                    a=m(X,M);b=m(X.masked_fill(~M[:,:,None],5.75),M)
                    padding=float(abs(a-b).max().cpu());assert padding<1e-5,padding
                    separate=[]
                    for j in range(3):
                        n=int(M[j].sum().item());separate.append(float(abs(a[j,:n]-m(X[j:j+1,:n],M[j:j+1,:n])[0]).max().cpu()))
                    assert max(separate)<2e-5,separate
                records.append(dict(fold=f,kind=kind,seed=seed,prediction_max_error=error,padding_mutation_error=padding,
                    batch_vs_single_error=max(separate),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for j,i in enumerate(valid):
            g=groups[i];z=g.select('pair_id','hand_id')
            for kind,prs in predictions.items():
                inc=np.mean([s.base.conditioned(p[j][:,1:],mins[g['behavior_family'][0]]) for p in prs],axis=0)
                score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
                z=z.with_columns(pl.Series(kind,score),pl.Series(kind+'_inclusion',inc))
            parts.append(z)
    replay=pl.concat(parts).sort('pair_id','hand_id');saved=pl.read_parquet(s.ROOT/'oof.parquet').sort('pair_id','hand_id')
    columns=['independent','contextual','independent_inclusion','contextual_inclusion']
    assert replay.select('pair_id','hand_id').equals(saved.select('pair_id','hand_id'))
    error=float(abs(replay.select(columns).to_numpy()-saved.select(columns).to_numpy()).max());assert error<1e-7
    report=dict(models=records,hand_rows=len(saved),aggregate_max_error=error,outer_label_mutations_pass=True,
        limitations='Same input/objective/schedule; contextual arm has extra attention parameters. Reused folds give fixed-prediction comparisons, not fresh holdout evidence.')
    (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def evaluate():
    d=pl.read_parquet(s.ROOT/'oof.parquet').join(pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33'),'pressure59','full'),on=['pair_id','hand_id'],validate='1:1')
    d=d.with_columns(*[((C(k)+C('pressure59'))*.5).alias(k+'_recipe') for k in s.CONFIG['kinds']]);path=s.ROOT/'comparison_oof.parquet';d.write_parquet(path)
    names=['r33','full','independent','contextual','independent_recipe','contextual_recipe'];r,_=compare(path,names,'session118')
    pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)))
    report={}
    for k in names:
        diff=pool[k].to_numpy()-pool['r33'].to_numpy();boot=diff[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
        report[k]=dict(MAP=r[k].mean(),delta_vs_r33=r[k].mean()-r['r33'].mean(),ci95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist(),
            folds=r.group_by('fold').agg(C(k).mean()).sort('fold')[k].to_list(),families=dict(r.group_by('family').agg(C(k).mean()).iter_rows()))
    (s.ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    import sys
    {'verify':verify,'compare':evaluate}[sys.argv[1]]()
