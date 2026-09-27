import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');sys.path.insert(0,'src/policy')
from pathlib import Path
import json
import numpy as np,polars as pl
from session4_evidence_model import correct
ROOT=Path('artifacts/evidence_session5');OLD=Path('artifacts/evidence_session4');C=pl.col
d=pl.read_parquet('artifacts/policy/evidence_training.parquet').join(pl.read_parquet(OLD/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
d=d.join(pl.read_parquet(OLD/'leafwise_separate_oof.parquet').select('pair_id','hand_id',C('score').alias('hgb')),on=['pair_id','hand_id'])
variants={}
for mode in ['ordinal','multiway','action','mil','fullbag','relative','censored','pretrained']:
    path=ROOT/f'{mode}_oof.parquet'
    if not path.exists():continue
    q=pl.read_parquet(path)
    for col in ['score','graded_score','noisyor_score','first5_score']:
        if col not in q.columns:continue
        name=mode if col=='score' else mode+'_'+col.removesuffix('_score');variants[name]=name
        d=d.join(q.select('pair_id','hand_id',C(col).alias(name)),on=['pair_id','hand_id'],validate='1:1')
prior=pl.read_csv(OLD/'nested_blend/set_validation.csv').select('pair_id',C('unary').alias('r27'))
rows=[]
weights=[np.load(OLD/f'nested_blend/unary_weights_fold{f}.npy') for f in range(4)]
for (pid,),g in d.group_by('pair_id'):
    truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth))
    row={'pair_id':pid,'table_id':g['table_id'][0],'fold':g['fold'][0],'family':g['behavior_family'][0]}
    for n in variants:
        for mode in ['raw','corrected','replace_cat']:
            col=n
            if mode=='raw':hands=g.sort([n,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5]
            else:
                expr=C(n) if mode=='corrected' else .5*C(n)+.5*C('hgb')
                hands=correct(g.with_columns(expr.alias('base_score')),weights[g['fold'][0]])
            y=np.array([h in truth for h in hands]);row[n+'_'+mode]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
    rows.append(row)
r=pl.DataFrame(rows).join(prior,on='pair_id');r.write_csv(ROOT/'comparison.csv');names=[n for n in r.columns if n not in ['pair_id','table_id','fold','family']]
pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n'));rng=np.random.default_rng(510);ix=rng.integers(0,len(pool),(3000,len(pool)));den=pool['n'].to_numpy()[ix].sum(1);stats={}
for n in names:
    delta=pool[n].to_numpy()-pool['r27'].to_numpy();boot=delta[ix].sum(1)/den
    stats[n]={'map5':r[n].mean(),'gain_vs_r27':r[n].mean()-r['r27'].mean(),'ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist(),
              'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),
              'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows())}
(ROOT/'comparison.json').write_text(json.dumps(stats,indent=2))
print(json.dumps({n:round(v['map5'],7) for n,v in stats.items()},indent=2))
