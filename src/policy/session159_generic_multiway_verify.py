import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
import numpy as np,polars as pl
import session158_generic_multiway_features as s

C=pl.col
ORIGINAL_READ=pl.read_parquet

def mutated_read(source,*args,**kwargs):
    q=ORIGINAL_READ(source,*args,**kwargs)
    if 'compact/seats/' in str(source) and 'net_chips' in q.columns:
        q=q.with_columns((-C('net_chips')+137).alias('net_chips'))
    return q

def design(table,local,query):
    raw=s.mw.build(table,query,return_actions=True)
    raw=raw.join(local.select('pair_id','hand_id','hand_index'),on=['pair_id','hand_id'],validate='m:1')
    return s.aggregate(local,raw)[0]

def main():
    d=pl.read_parquet(s.base.ROOT/'hands.parquet');old=np.load(s.ROOT/'extra.npy',mmap_mode='r')
    players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');checks=[]
    for fold in range(4):
        table=sorted(d.filter(C('fold')==fold)['table_id'].unique())[0]
        local=d.filter(C('table_id')==table).sort('hand_index').rename({'hand_index':'global_hand_index'}).with_row_index('hand_index')
        query=local.select('pair_id','hand_id').join(players,on='pair_id',validate='m:1')
        x=design(table,local,query);expected=old[local['global_hand_index'].to_numpy()]
        replay=float(abs(x-expected).max());assert replay==0
        swapped=query.select('pair_id','hand_id',C('player_2').alias('player_1'),C('player_1').alias('player_2'))
        role=float(abs(design(table,local,swapped)-x).max());assert role==0
        try:
            pl.read_parquet=mutated_read
            outcome=float(abs(design(table,local,query)-x).max());assert outcome==0
        finally:pl.read_parquet=ORIGINAL_READ
        checks.append(dict(table=table,fold=fold,hands=len(local),raw_replay_error=replay,player_role_error=role,outcome_mutation_error=outcome))
    out=dict(checks=checks,scope='Four development pools: raw calculator/aggregate replay, swapped query endpoints and mutated final chip outcomes. No model, validation improvement or future-board mutation claim.')
    (s.ROOT/'raw_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
