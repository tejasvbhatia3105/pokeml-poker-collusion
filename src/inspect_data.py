from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
for p in sorted(Path('data').glob('*')):
    if p.suffix == '.parquet':
        f=pq.ParquetFile(p)
        print('\nFILE',p.name,'ROWS',f.metadata.num_rows,'SCHEMA',f.schema_arrow)
        print(next(f.iter_batches(batch_size=3)).to_pandas().to_string(index=False))
    elif p.suffix == '.csv':
        d=pd.read_csv(p)
        print('\nFILE',p.name,d.shape,d.head().to_string(index=False))
        if 'label' in p.name:
            for c in d.columns:
                print(c, d[c].value_counts().head(20).to_dict())
