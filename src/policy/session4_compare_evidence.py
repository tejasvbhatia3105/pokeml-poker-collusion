import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import polars as pl,numpy as np,json
OUT=Path('artifacts/evidence_session4');C=pl.col
d=pl.read_parquet('artifacts/policy/relationship_evidence/oof.parquet').rename({'score':'production'})
ix=pl.read_parquet(OUT/'hand_index.parquet').select('pair_id','hand_id','table_id','fold')
d=d.join(ix,on=['pair_id','hand_id'])
base=pl.concat([pl.read_parquet(OUT/f'nested_outer{f}.parquet').join(ix.filter(C('fold')==f).select('pair_id','hand_id'),on=['pair_id','hand_id']) for f in range(4)])
d=d.join(base,on=['pair_id','hand_id']);names=['production','base_score']
for n in ['attribution','leafwise_separate','leafwise_pooled','raw_only','raw_combined']:
    path=OUT/f'{n}_oof.parquet'
    if not path.exists():continue
    d=d.join(pl.read_parquet(path).select('pair_id','hand_id',C('score').alias(n)),on=['pair_id','hand_id'],validate='1:1')
    d=d.with_columns(((C(n)+C('base_score'))/2).alias(n+'_half_base'),((C(n)+C('production'))/2).alias(n+'_half_production'))
    names += [n,n+'_half_base',n+'_half_production']
rows=[]
for (pid,),g in d.group_by('pair_id'):
    row=dict(pair_id=pid,table_id=g['table_id'][0],fold=g['fold'][0],family=g['behavior_family'][0])
    den=min(5,g['evidence'].sum())
    for n in names:
        y=g.sort([n,'hand_id'],descending=[True,False])['evidence'].to_numpy()[:5]
        row[n]=float(np.sum(y*np.cumsum(y)/np.arange(1,len(y)+1))/den)
    rows.append(row)
r=pl.DataFrame(rows);r.write_csv(OUT/'comparison.csv')
print(r.group_by('family').agg(C(names).mean()))
print(json.dumps(r.select(C(names).mean()).to_dicts()[0],indent=2))
pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n'));rng=np.random.default_rng(410)
ix=rng.integers(0,len(pool),size=(3000,len(pool)));den=pool['n'].to_numpy()[ix].sum(1);stats={}
for n in names:
    stats[n]={'map5':r[n].mean(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list()}
    for ref in ['base_score','production']:
        delta=pool[n].to_numpy()-pool[ref].to_numpy();boot=delta[ix].sum(1)/den
        stats[n]['vs_'+ref]={'gain':r[n].mean()-r[ref].mean(),'ci95':np.quantile(boot,[.025,.975]).tolist()}
(OUT/'comparison.json').write_text(json.dumps(stats,indent=2))
