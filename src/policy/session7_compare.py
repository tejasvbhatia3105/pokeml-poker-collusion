import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
ROOT=Path('artifacts/evidence_session7');OLD=Path('artifacts/evidence_session4');C=pl.col
def frame():
    d=pl.read_parquet('artifacts/policy/evidence_training.parquet').join(pl.read_parquet(OLD/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'],validate='1:1')
    prior=pl.concat([pl.read_parquet(OLD/f'nested_blend/nested_outer{f}.parquet').join(d.select('pair_id','hand_id','fold'),on=['pair_id','hand_id']).filter(C('fold')==f).select('pair_id','hand_id',C('base_score').alias('r27_base')) for f in range(4)])
    d=d.join(prior,on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet').select('pair_id','hand_id',C('score').alias('r28_priority')),on=['pair_id','hand_id'],validate='1:1')
    return d.with_columns(((C('r27_base')+C('r28_priority'))*.5).alias('r28'))
def main():
    d=frame();names=['r28'];rows=[]
    for path in sorted(ROOT.glob('*_oof.parquet')):
        name=path.stem.removesuffix('_oof');q=pl.read_parquet(path)
        d=d.join(q.select('pair_id','hand_id',C('score').alias(name)),on=['pair_id','hand_id'],validate='1:1')
        d=d.with_columns(((C(name)+C('r27_base'))*.5).alias(name+'_half_base'),((C(name)+C('r28_priority'))*.25+C('r27_base')*.5).alias(name+'_quarter_replace'),((C(name)+C('r28'))*.5).alias(name+'_half_r28'));names.extend([name,name+'_half_base',name+'_quarter_replace',name+'_half_r28'])
    for (pid,),g in d.group_by('pair_id'):
        truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth));row={'pair_id':pid,'table_id':g['table_id'][0],'fold':g['fold'][0],'family':g['behavior_family'][0]}
        for name in names:
            hands=g.sort([name,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([h in truth for h in hands]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
        rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'comparison.csv');pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(710).integers(0,len(pool),(3000,len(pool)));stats={}
    for name in names:
        delta=pool[name].to_numpy()-pool['r28'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
        stats[name]={'map5':r[name].mean(),'gain':r[name].mean()-r['r28'].mean(),'ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(name).mean()).sort('fold')[name].to_list(),'families':dict(r.group_by('family').agg(C(name).mean()).iter_rows())}
    (ROOT/'comparison.json').write_text(json.dumps(stats,indent=2));print(json.dumps({n:v['map5'] for n,v in stats.items()},indent=2))
if __name__=='__main__':main()
