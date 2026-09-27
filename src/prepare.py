from pathlib import Path
import duckdb,time
c=duckdb.connect(); c.execute("SET threads=6; SET memory_limit='5GB'; SET temp_directory='artifacts/tmp'")
for name in ['hands','seats','actions']:
    if Path(f'artifacts/partitioned/{name}').exists(): continue
    t=time.time(); print('partition',name,flush=True)
    if name=='hands': q="SELECT * FROM read_parquet('data/hands.parquet')"
    else: q=f"SELECT s.*,h.table_id FROM read_parquet('data/{name}.parquet') s JOIN read_parquet('data/hands.parquet') h USING(hand_id)"
    c.execute(f"COPY ({q}) TO 'artifacts/partitioned/{name}' (FORMAT PARQUET, PARTITION_BY(table_id), COMPRESSION ZSTD)")
    print('seconds',time.time()-t,flush=True)
