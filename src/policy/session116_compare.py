import itertools,json
import numpy as np
import polars as pl
from session8_data import hand_data
from session12_compare import compare
from session116_relationship_encoder import ROOT
C=pl.col

def main():
    d=hand_data().select('pair_id','hand_id','fold').with_row_index('hand_row');values=np.zeros((len(d),2));seen=np.zeros(len(d),int)
    for f,g in itertools.combinations(range(4),2):
        z=d.filter(C('fold').is_in([f,g])).join(pl.read_parquet(ROOT/f'exclude{f}{g}.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
        assert z['row'].null_count()==0;ix=z['hand_row'].to_numpy();values[ix]+=z.select('local','context').to_numpy()/3;seen[ix]+=1
    assert (seen==3).all()
    out=d.select('pair_id','hand_id').with_columns(pl.Series('local',values[:,0]),pl.Series('context',values[:,1]))
    baseline=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33'))
    out.join(baseline,on=['pair_id','hand_id'],validate='1:1').write_parquet(ROOT/'evidence_oof.parquet')
    r,_=compare(ROOT/'evidence_oof.parquet',['r33','local','context'],'session116')
    report={name:dict(MAP=r[name].mean(),folds=r.group_by('fold').agg(C(name).mean()).sort('fold')[name].to_list(),families=dict(r.group_by('family').agg(C(name).mean()).iter_rows())) for name in ['r33','local','context']}
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
