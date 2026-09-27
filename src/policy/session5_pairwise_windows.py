import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from sequence_features import augment
from session4_evidence_model import load_models,score,COLS,ROOT,FAMILIES
from session5_pairwise_model import load_models as load_comparator,rankings
models=load_models();backends=['cat','hist','residual','context'];comparators={b:load_comparator(b) for b in backends}
ix=pl.read_parquet(ROOT/'hand_index.parquet');data=pl.read_parquet('artifacts/evidence_windows/window_features.parquet').join(ix.select('pair_id','hand_id','time','fold','net_direction'),on=['pair_id','hand_id'])
rows=[];C=pl.col
with threadpool_limits(limits=4):
    for window,lo,hi in [('first_2000',0,2000),('last_2000',1000,3000)]:
        z=data.filter(C('window')==window).with_columns(pl.lit('development').alias('phase'),((C('time')*5000-lo)/(hi-lo)).alias('relative_time'));z,_=augment(z)
        for f in range(4):
            weights=np.load(ROOT/f'nested_blend/unary_weights_fold{f}.npy')
            for family in FAMILIES:
                q=z.filter((C('fold')==f)&(C('behavior_family')==family))
                if not len(q):continue
                q=q.with_columns(pl.Series('base_score',score(models,family,q.select(COLS).to_numpy(),[f])))
                for (pid,),g in q.group_by('pair_id'):
                    true=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(true))
                    if not den:continue
                    row={'pair_id':pid,'table_id':g['table_id'][0],'fold':f,'window':window,'family':family}
                    for backend in backends:
                        ranks=rankings(g,weights,comparators[backend],[f],backend)
                        for name,hands in ranks.items():
                            y=np.array([h in true for h in hands]);row[backend+'_'+name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
                    rows.append(row)
r=pl.DataFrame(rows);r.write_csv('artifacts/evidence_session5/pairwise_windows.csv');names=[n for n in r.columns if n not in ['pair_id','table_id','fold','window','family']]
print(r.group_by('window').agg(pl.len(),C(names).mean()))
Path('artifacts/evidence_session5/pairwise_windows.json').write_text(json.dumps(r.group_by('window','family').agg(pl.len(),C(names).mean()).to_dicts(),indent=2))
