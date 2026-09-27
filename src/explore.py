import duckdb
c=duckdb.connect('artifacts/poker.duckdb'); c.execute("SET threads=6; SET memory_limit='6GB'; SET temp_directory='artifacts/tmp'")
for name in ['hands','seats','actions']:
    c.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('data/{name}.parquet')")
c.execute("CREATE OR REPLACE TABLE labels AS SELECT * FROM read_csv_auto('data/development_labels.csv')")
c.execute("CREATE OR REPLACE TABLE evidence AS SELECT * FROM read_csv_auto('data/development_evidence.csv')")
for behavior in ['directed_transfer','soft_play','coordinated_isolation']:
    e=c.sql(f"select e.*,l.player_1,l.player_2 from evidence e join labels l using(pair_id) where e.behavior_family='{behavior}' limit 3").df()
    for r in e.itertuples():
        print('\nEVIDENCE',behavior,r.pair_id,r.hand_id,r.player_1,r.player_2)
        print(c.sql(f"select * from hands where hand_id='{r.hand_id}'").df().to_string(index=False))
        print(c.sql(f"select * from seats where hand_id='{r.hand_id}'").df().to_string(index=False))
        print(c.sql(f"select * from actions where hand_id='{r.hand_id}' order by action_no").df().to_string(index=False))
