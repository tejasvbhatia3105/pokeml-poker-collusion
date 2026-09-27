import hashlib,json
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
import session103_nested_sizes as s

def main():
    d=pl.read_parquet(sorted(Path('artifacts/evidence_session22_size_density/samples').glob('T*.parquet')))
    fold=d['fold'].to_numpy();values=d['log_bet_ratio'].to_numpy();allin=d['allin'].to_numpy();records=[]
    tf=json.load(open('artifacts/policy/table_folds.json'))
    np.testing.assert_array_equal(fold,np.array([tf[t] for t in d['table_id']]))
    for f in range(4):
        for h in range(f+1,4):
            tr=(fold!=f)&(fold!=h);meta=json.load(open(s.ROOT/f'metadata_exclude{f}{h}.json'))
            def reconstruct(size):
                v=size[tr&~allin];edges=np.unique(np.quantile(v,np.linspace(0,1,17)))
                edges[0]=min(-8.,float(v.min())-.1);edges[-1]=max(8.,float(v.max())+.1)
                labels=np.searchsorted(edges,size,side='right')-1;labels=np.clip(labels,0,len(edges)-2);labels[allin]=len(edges)-1
                centers=np.array([size[tr&(labels==k)].mean() for k in range(len(edges))])
                return edges,centers,labels[tr]
            edges,centers,y=reconstruct(values);mut=values.copy();mut[~tr]=999.
            other=reconstruct(mut)
            for a,b in zip((edges,centers,y),other):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(edges,meta['edges']);np.testing.assert_array_equal(centers,meta['centers'])
            assert sorted(d.filter(pl.Series(tr))['table_id'].unique().to_list())==meta['training_tables']
            assert all(tf[t] not in [f,h] for t in meta['training_tables'])
            path=s.ROOT/f'size_exclude{f}{h}.cbm';assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==meta['model_sha256']
            model=CatBoostClassifier();model.load_model(str(path));assert model.tree_count_==400 and list(model.classes_)==list(range(len(edges)))
            records.append({'excluded_folds':[f,h],'bins_and_centers_exact':True,'excluded_size_mutation_error':0,'training_tables_verified':len(meta['training_tables'])})
    out={'models':6,'records':records,'source_fold_mapping_exact':True,'model_hashes_verified':True}
    (s.ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
