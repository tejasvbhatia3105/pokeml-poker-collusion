import json,hashlib
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
import session123_family_holdout as s
C=pl.col

def main():
    import sys
    control_only=len(sys.argv)>1 and sys.argv[1]=='control-only'
    d,x,groups=s.load();records=[];fields=['score','primary','secondary'];maxerror=0.
    for held in (['all'] if control_only else ['all']+s.FAMILIES):
        for fold in range(4):
            tr,va=s.masks(d,fold,held);prob=s.initial(d,groups);protected=~tr
            mutated=d.with_columns(pl.when(pl.Series(protected)).then(pl.lit(999)).otherwise(C('evidence_rank')).alias('evidence_rank'))
            for step in range(s.CONFIG['em_steps']):
                a=s.targets(d,groups,prob,tr);b=s.targets(mutated,groups,prob,tr)
                for first,second in zip(a[:4],b[:4]):np.testing.assert_array_equal(first,second)
                assert a[4:]==b[4:];rows,cat,w,used,minimum,excluded=a
                assert not np.any(used&protected)
                if held!='all':assert not np.any((d['behavior_family'].to_numpy()[rows])==held)
                path=s.ROOT/held/f'fold{fold}_em{step+1}.cbm';record=json.load(open(path.with_suffix('.json')))
                assert record['target_sha256']==s.checksum(rows,cat,w)
                assert record['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                assert set(record['training_tables']).isdisjoint(d.filter(C('fold')==fold)['table_id'])
                model=CatBoostClassifier();model.load_model(str(path));assert model.tree_count_==400
                prob=model.predict_proba(x,thread_count=3)
                sample=np.flatnonzero(va)[::97];reverse=model.predict_proba(x[sample[::-1]],thread_count=3)[::-1]
                assert np.array_equal(prob[sample],reverse)
                records.append(dict(held_family=held,fold=fold,step=step+1,target_mutation_exact=True,training_target_hash_exact=True,query_reversal_exact=True))
            parts=[]
            for ix in groups:
                if not va[ix[0]]:continue
                parts.append(d[ix].select('hand_index','pair_id','hand_id').with_columns(pl.Series('score',s.conditioned(prob[ix,1:],minimum)),pl.Series('primary',prob[ix,1]),pl.Series('secondary',prob[ix,2])))
            replay=pl.concat(parts).sort('hand_index');saved=pl.read_parquet(s.ROOT/held/f'fold{fold}.parquet').sort('hand_index')
            assert replay.select('pair_id','hand_id').equals(saved.select('pair_id','hand_id'))
            error=float(abs(replay.select(fields).to_numpy()-saved.select(fields).to_numpy()).max());assert error==0;maxerror=max(maxerror,error)
    teacher_root=s.CONFIG.get('teacher_root')
    ancestry=('Training initialization uses nested supervised teachers; their exclusions are checked separately in preparation_verification.json and teacher_manifest.json.'
              if teacher_root else 'No trained teacher ancestry.')
    report=dict(models=len(records),scope='all-family control only' if control_only else 'full family-held-out benchmark',records=records,final_prediction_error=maxerror,pool_exclusions=True,family_exclusions_tested=not control_only,
        teacher_root=teacher_root,query_reversal_scope='Reverse prediction row order; not player-role reversal. Raw player-role replay is a separate inference check.',
        limitations=ancestry+' Feature design and repeated known-family data are not an untouched blind evaluation. Only conditional evidence retrieval is measured.')
    (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
