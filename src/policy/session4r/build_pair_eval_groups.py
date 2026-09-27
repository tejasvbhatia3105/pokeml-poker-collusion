\
\
\
import sys,numpy as np,polars as pl
from pathlib import Path
C=pl.col; N=['directed_transfer','soft_play','coordinated_isolation']
out=Path(sys.argv[1]); out.mkdir(exist_ok=True,parents=True); spec=sys.argv[2]
groups=[(g.split(':')[0].split(','),float(g.split(':')[1]) if ':' in g else 1.0) for g in spec.split('|')]
def lg(v): return pl.read_csv(f'artifacts/{v}/eval_all_lpo.csv').select('pair_id',(1-C('none')).clip(0,1).log().alias(v))
d=pl.read_csv('artifacts/candidate_r23/pair_eval.csv').select('pair_id',(1-C('none')).alias('r23'),*[C(n).alias(n+'_23') for n in N])
d=d.join(pl.read_csv('artifacts/candidate_r12/pair_eval.csv').select('pair_id',(1-C('none')).alias('r12')),on='pair_id')
tot=0; acc=None
for grp,w in groups:
    g=lg(grp[0])
    for v in grp[1:]: g=g.join(lg(v),on='pair_id')
    g=g.select('pair_id',pl.mean_horizontal([c for c in g.columns if c!='pair_id']).alias('m'))
    d=d.join(g,on='pair_id'); acc=(w*C('m')) if acc is None else acc+w*C('m'); tot+=w; d=d.with_columns(acc.alias('acc')).drop('m'); acc=C('acc')
d=d.with_columns((C('acc')/tot).exp().alias('seq')).with_columns((C('r12')*C('seq')).sqrt().alias('risk'))
for n in N: d=d.with_columns(pl.when(C('r23')>0).then(C(n+'_23')/C('r23')).otherwise(1/3).alias(n+'_c'))
s=sum(C(n+'_c') for n in N); d=d.with_columns(*[(C(n+'_c')/s*C('risk')).alias(n) for n in N],(1-C('risk')).alias('none'))
d.select('pair_id','none',*N).write_csv(out/'pair_eval.csv'); print('rows',d.height,'risk>=0.5',int((d['risk']>=0.5).sum()),'>=0.05',int((d['risk']>=0.05).sum()),'>=0.01',int((d['risk']>=0.01).sum()))
for cmp in ['artifacts/candidate_r26/pair_eval.csv']:
    c=pl.read_csv(cmp).select('pair_id',(1-C('none')).alias('rc')); j=d.join(c,on='pair_id'); print('max |risk - r26|',float((j['risk']-j['rc']).abs().max()),'top500 overlap',len(set(j.sort('risk',descending=True)['pair_id'][:500])&set(j.sort('rc',descending=True)['pair_id'][:500])))
