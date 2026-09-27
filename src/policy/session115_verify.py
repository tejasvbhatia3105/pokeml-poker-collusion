import json
from pathlib import Path
import numpy as np
import polars as pl
import session115_relationship_data as s
from session8_data import hand_data
C=pl.col

def main():
    cfg=json.loads((s.ROOT/'config.json').read_text());x=np.load(s.ROOT/'x.npy',mmap_mode='r');b=pl.read_parquet(s.ROOT/'bags.parquet');h=pl.read_parquet(s.ROOT/'hands.parquet')
    t1,t2=map(Path,cfg['source_tokens']);rebuilt=0
    for (table,),local in b.group_by('table_id'):
        with np.load(t1/f'{table}.npz') as z:
            source1=np.load(t1/f'{table}_X.npy',mmap_mode='r');source2=np.load(t2/f'{table}.npy',mmap_mode='r');offset=np.r_[0,np.cumsum(z['n'])]
            ix={str(pid):i for i,(pid,phase) in enumerate(zip(z['pair_id'],z['phase'])) if phase=='development'}
            for row in local.to_dicts():
                i=ix[row['pair_id']];a,c=map(int,offset[i:i+2]);start,n=row['offset'],row['n_hands'];assert n==c-a
                expected=np.column_stack([source1[a:c],source2[a:c]]).astype(np.float32)
                np.testing.assert_array_equal(x[start:start+n],expected)
                np.testing.assert_array_equal(h['hand_id'].slice(start,n).to_numpy(),z['hand_id'][a:c])
                np.testing.assert_array_equal(h['time_index'].slice(start,n).to_numpy(),z['time_index'][a:c]);rebuilt+=n
    truth=set(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').iter_rows())
    y=np.array([(pid,hid) in truth for pid,hid in h.select('pair_id','hand_id').iter_rows()],np.uint8)
    np.testing.assert_array_equal(y,h['evidence'].to_numpy());assert int(y.sum())==1817
    current=hand_data().select('pair_id','hand_id','fold');joined=current.join(h.select('pair_id','hand_id','row'),on=['pair_id','hand_id'],how='left',validate='1:1')
    assert joined['row'].null_count()==0
    actual=current.select('pair_id','fold').unique().join(b.select('pair_id',C('fold').alias('sequence_fold')),on='pair_id',validate='1:1')
    assert (actual['fold']==actual['sequence_fold']).all()
    plans=json.loads((s.ROOT/'reference_plan.json').read_text());mutations=0
    for plan in plans:
        excluded=plan['excluded_folds'];tr=b.filter(~C('fold').is_in(excluded));held=b.filter(C('fold').is_in(excluded))
        assert set(tr['pair_id'])==set(plan['training_pair_ids']) and not set(tr['table_id'])&set(held['table_id'])
        original=h.join(tr.select('pair_id'),on='pair_id',how='semi').sort('row')
        changed=h.with_columns(pl.when(C('pair_id').is_in(held['pair_id'].to_list())).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'))
        after=changed.join(tr.select('pair_id'),on='pair_id',how='semi').sort('row');np.testing.assert_array_equal(original.to_numpy(),after.to_numpy());mutations+=1
    lost=0;positive_cropped=0
    for row in b.filter(C('label')==1).to_dicts():
        excess=max(0,row['n_hands']-160);lost+=int(y[row['offset']:row['offset']+excess].sum());positive_cropped+=int(excess>0)
    report=dict(cached_hand_rows_rebuilt=rebuilt,all_feature_values_exact=True,pair_specific_labels_rebuilt=len(y),true_evidence_hands=1817,
        current_R33_positive_hand_queries_covered=len(current),current_fold_mapping_exact=True,heldout_label_mutations=mutations,
        historical_last160_would_drop_true_evidence=lost,positive_pairs_exceeding160_hands=positive_cropped,
        caveat='Truncation count is for this full trusted development dataset, not a claim about R33, which uses a different evidence pipeline.')
    (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
