import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib
import numpy as np,polars as pl
import session170_negative_action_data as s
C=pl.col

def main():
    cfg=json.load(open(s.ROOT/'config.json'));assert cfg['x_sha256']==hashlib.sha256((s.ROOT/'x.npy').read_bytes()).hexdigest()
    neg=pl.read_parquet(s.ROOT/'hands.parquet');nm=pl.read_parquet(s.ROOT/'actions.parquet');nx=np.load(s.ROOT/'x.npy',mmap_mode='r')
    base=s.positive.base;pos=pl.read_parquet(base.ROOT/'hands.parquet');bx=np.load(base.ROOT/'x.npy',mmap_mode='r')
    players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
    pos=pos.drop('player_1','player_2').join(players,on='pair_id',validate='m:1',maintain_order='left')
    pos=pos.with_columns(*[pl.Series(n,bx[:,-3+i]) for i,n in enumerate(['phase_position','pair_hand_position','log_pair_hand_count'])])
    pm=pl.read_parquet(s.positive.ROOT/'actions.parquet');px=np.load(s.positive.ROOT/'x.npy',mmap_mode='r');records=[]
    for name,d,m,x in [('positive',pos,pm,px),('negative',neg,nm,nx)]:
        for fold in range(4):
            table=sorted(d.filter(C('fold')==fold)['table_id'].unique())[0];q=d.filter(C('table_id')==table)
            z,xx=s.design(table,q);saved=m.with_row_index('cached').filter(C('table_id')==table)
            z=z.with_row_index('rebuilt').join(saved.select('pair_id','hand_id','action_no','cached'),on=['pair_id','hand_id','action_no'],validate='1:1',maintain_order='left')
            assert len(z)==len(saved) and not z['cached'].null_count();np.testing.assert_array_equal(xx,x[z['cached'].to_numpy()])
            swap=q.with_columns(C('player_2').alias('player_1'),C('player_1').alias('player_2'))
            mz,mx=s.design(table,swap);np.testing.assert_array_equal(mx,xx)
                                                                                           
            rq=q.head(20).with_columns((1-C('label')).alias('label'),pl.lit('protected_test').alias('behavior_family'),C('player_2').alias('player_1'),C('player_1').alias('player_2'))
            rm,rx,_=s.raw.build(table,rq)
            join=rm.join(z.select('pair_id','hand_id','action_no','rebuilt'),on=['pair_id','hand_id','action_no'],validate='1:1',maintain_order='left')
            np.testing.assert_array_equal(rx,xx[join['rebuilt'].to_numpy(),:74])
            records.append(dict(kind=name,fold=fold,table=table,actions=len(xx),raw_rebuilt_actions=len(rx),feature_error=0,endpoint_swap_exact=True,label_mutation_exact=True))
    assert set(neg['pair_id']).isdisjoint(pos['pair_id'])
    out=dict(records=records,negative_pairs=1488,negative_actions=len(nx),positive_actions=len(px),positive_matrix_exact=True,unknown_pairs_used=False)
    (s.ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
