import json
import numpy as np
import polars as pl
import session119_private_partner_data as s
C=pl.col

def main():
    d=s.source();records=[]
    for fold in range(4):
        table=sorted(set(d.filter(C('fold')==fold)['table_id']))[0];q=d.filter(C('table_id')==table).sort('hand_id','pair_id')
        saved=pl.read_parquet(s.ROOT/f'{table}.parquet');x=np.load(s.ROOT/f'{table}.npz')['x']
        z,replay,_=s.build(table,q);assert saved.equals(z) and np.array_equal(x,replay)
                                                                            
                                                                           
        mut=q.with_columns((1-C('label')).alias('label'),pl.lit('unseen_test_family').alias('behavior_family'),
            C('player_2').alias('player_1'),C('player_1').alias('player_2'))
        mz,mx,_=s.build(table,mut)
        assert np.array_equal(x,mx) and np.array_equal(z['actor'].to_numpy(),1-mz['actor'].to_numpy())
        raw,meta,seats=s.engine.table_data(table,q.select('hand_id').unique())
        assert all(v['phase']=='development' for v in meta.values())
        rel={p:(a,b) for p,a,b in q.select('pair_id','player_1','player_2').unique().iter_rows()}
                                                                             
                                                                              
        histories={h:g.to_dicts() for (h,),g in raw.group_by('hand_id')};error=0.;checked=0
        for i in range(0,len(z),max(1,len(z)//64)):
            row=z.row(i,named=True);aa,bb=rel[row['pair_id']];own,partner=(aa,bb) if row['actor']==0 else (bb,aa)
            prev=[r for r in histories[row['hand_id']] if r['action_no']<row['action_no']]
            groups=[[r for r in prev if r['player_id']==own],[r for r in prev if r['player_id']==partner],
                [r for r in prev if r['player_id'] not in [own,partner]]]
            counts=[sum(s.engine.action_class(r)==k for r in g) for g in groups for k in range(4)]
            ix=[s.PUBLIC.index(f'{who}_prior_{k}') for who in ['own','partner','others'] for k in range(4)]
            error=max(error,float(abs(x[i,ix]-counts).max()))
            last=max(groups[1],key=lambda r:r['action_no']) if groups[1] else None
            assert x[i,s.PUBLIC.index('partner_last_action')]==(-1 if last is None else s.engine.action_class(last))
            amount=sum(r['amount'] for r in groups[1])/meta[row['hand_id']]['big_blind']
            assert abs(x[i,s.PUBLIC.index('partner_prior_amount_bb')]-amount)<1e-4
            checked+=1
        assert error==0
        records.append(dict(fold=fold,table=table,replayed_actions=len(z),role_swap_and_label_mutation_exact=True,
            independent_prefix_samples=checked,prefix_count_error=error))
    report=dict(records=records,all397_pool_build_checks=json.load(open(s.ROOT/'audit.json')),
        caveat='Static card features may be read from a later action on the same street, but depend only on private holdings and the already visible board; future-card masking checks pass. Style counts use other hands in the same phase.')
    (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(records,indent=2))

if __name__=='__main__':main()
