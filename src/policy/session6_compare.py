import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from session4_evidence_model import correct
ROOT=Path('artifacts/evidence_session6');OLD=Path('artifacts/evidence_session4');C=pl.col
d=pl.read_parquet('artifacts/policy/evidence_training.parquet').join(pl.read_parquet(OLD/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
prior=pl.concat([pl.read_parquet(OLD/f'nested_blend/nested_outer{f}.parquet').join(d.select('pair_id','hand_id','fold'),on=['pair_id','hand_id']).filter(C('fold')==f).select('pair_id','hand_id',C('base_score').alias('r27_base')) for f in range(4)])
d=d.join(prior,on=['pair_id','hand_id'],validate='1:1')
modes=[p.stem.removesuffix('_oof') for p in ROOT.glob('*_oof.parquet')];scores=[]
for mode in modes:
    x=pl.read_parquet(ROOT/f'{mode}_oof.parquet');names=[n for n in ['cat','hist','score','uncapped'] if n in x.columns]
    d=d.join(x.select('pair_id','hand_id',*[C(n).alias(mode+'_'+n) for n in names]),on=['pair_id','hand_id'],validate='1:1');scores.extend(mode+'_'+n for n in names)
    if 'cat' in names and 'hist' in names:d=d.with_columns(((C(mode+'_cat')+C(mode+'_hist'))/2).alias(mode+'_blend'));scores.append(mode+'_blend')
    if mode.startswith('priority'):
        d=d.with_columns(((C(mode+'_score')+C('r27_base'))/2).alias(mode+'_half_r27'));scores.append(mode+'_half_r27')
weights=[np.load(OLD/f'nested_blend/unary_weights_fold{f}.npy') for f in range(4)];rows=[]
for (pid,),g in d.group_by('pair_id'):
    truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth));row={'pair_id':pid,'table_id':g['table_id'][0],'fold':g['fold'][0],'family':g['behavior_family'][0]}
    for name in scores:
        for kind in ['raw','corrected']:
            hands=g.sort([name,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5] if kind=='raw' else correct(g.with_columns(C(name).alias('base_score')),weights[g['fold'][0]])
            y=np.array([h in truth for h in hands]);row[name+'_'+kind]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
    rows.append(row)
r=pl.DataFrame(rows).join(pl.read_csv(OLD/'nested_blend/set_validation.csv').select('pair_id',C('unary').alias('r27')),on='pair_id');r.write_csv(ROOT/'comparison.csv');names=[n for n in r.columns if n not in ['pair_id','table_id','fold','family']];pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(610).integers(0,len(pool),(3000,len(pool)));stats={}
for n in names:
    delta=pool[n].to_numpy()-pool['r27'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
    stats[n]={'map5':r[n].mean(),'gain':r[n].mean()-r['r27'].mean(),'ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows())}
(ROOT/'comparison.json').write_text(json.dumps(stats,indent=2));print(json.dumps({n:v['map5'] for n,v in stats.items()},indent=2))
