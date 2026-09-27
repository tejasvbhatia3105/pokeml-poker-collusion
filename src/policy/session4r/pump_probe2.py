\
\
import os; os.environ.setdefault('POLARS_MAX_THREADS','6')
import numpy as np,polars as pl,time
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
rows=[]; t0=time.time()
for i,p in enumerate(sorted(Path('artifacts/policy/actions').glob('*.parquet'))):
    t=p.stem
    a=pl.read_parquet(p,columns=['hand_id','player_id','phase','street_no','action_class','equity','action_no']).filter(C('street_no')==0)
    first=a.sort('action_no').group_by('hand_id','player_id',maintain_order=True).agg(C('action_class').first().alias('a'),C('equity').first().alias('eP'),C('action_no').first().alias('no'),C('phase').first())
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0).select('hand_id',C('player_id').alias('q'),C('equity').alias('eQ'))
    qno=first.select('hand_id',C('player_id').alias('q'),C('no').alias('qno'))
    j=first.join(st,on='hand_id').filter(C('q')!=C('player_id')).join(qno,on=['hand_id','q'],how='left').filter((C('qno')>C('no')).fill_null(True)).with_columns(pl.lit(t).alias('table'))
    rows.append(j.select('table','phase','player_id','q',(C('a')==3).cast(pl.Float64).alias('r'),'eP',(C('eQ')>=0.55).alias('sQ')))
    if i%100==0: print(i,round(time.time()-t0),flush=True)
d=pl.concat(rows); print('rows',d.height,flush=True)
d=d.with_columns((C('eP').rank('ordinal').over('phase')*20/pl.len().over('phase')).floor().clip(0,19).alias('bin'))
d=d.with_columns((C('r')-C('r').mean().over('phase','bin')).alias('rr'))
g=d.group_by('table','phase','player_id','q').agg(C('sQ').sum().alias('n_s'),(~C('sQ')).sum().alias('n_w'),C('r').filter(C('sQ')).mean().alias('r_s'),C('r').filter(~C('sQ')).mean().alias('r_w'),C('rr').filter(C('sQ')).mean().alias('rr_s'),C('rr').filter(~C('sQ')).mean().alias('rr_w'))
g=g.with_columns(((C('r_s')*C('n_s')+C('r_w')*C('n_w'))/(C('n_s')+C('n_w'))).alias('pbar')).with_columns(((C('r_s')-C('r_w'))/((C('pbar')*(1-C('pbar')))*(1/C('n_s')+1/C('n_w'))+1e-9).sqrt()).alias('z2'),(C('rr_s')-C('rr_w')).alias('lift_adj'))
g.write_parquet(S+'pump_probe2.parquet'); print('saved',g.shape,round(time.time()-t0),flush=True)
