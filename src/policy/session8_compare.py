import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import reference,ROOT,C
def main():
    d=reference();names=['r29'];rows=[]
    for path in sorted(ROOT.glob('*_oof.parquet')):
        name=path.stem.removesuffix('_oof');d=d.join(pl.read_parquet(path).select('pair_id','hand_id',C('score').alias(name)),on=['pair_id','hand_id'],validate='1:1')
        d=d.with_columns(((C(name)+C('r29'))*.5).alias(name+'_half_r29'),(C(name)*.25+C('r29')*.75).alias(name+'_quarter_r29'));names.extend([name,name+'_half_r29',name+'_quarter_r29'])
    for (pid,),g in d.group_by('pair_id'):
        truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth));row={'pair_id':pid,'table_id':g['table_id'][0],'fold':g['fold'][0],'family':g['behavior_family'][0]}
        for name in names:
            hand=g.sort([name,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([h in truth for h in hand]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
        rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'comparison.csv');pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(810).integers(0,len(pool),(3000,len(pool)));stats={}
    for name in names:
        delta=pool[name].to_numpy()-pool['r29'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
        stats[name]={'map5':r[name].mean(),'gain':r[name].mean()-r['r29'].mean(),'ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(name).mean()).sort('fold')[name].to_list(),'families':dict(r.group_by('family').agg(C(name).mean()).iter_rows())}
    (ROOT/'comparison.json').write_text(json.dumps(stats,indent=2));print(json.dumps({k:v['map5'] for k,v in stats.items()},indent=2))
if __name__=='__main__':main()
