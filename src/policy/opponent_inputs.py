import polars as pl
STYLES=[f'{p}_{k}' for p in ['style','local_style'] for k in range(4)]
COLUMNS=['aggressor_'+c for c in STYLES]
def enrich(a):
 profiles=a.group_by('hand_id','player_id','street_no').agg(*[pl.col(c).first().alias('aggressor_'+c) for c in STYLES]).rename({'player_id':'last_aggressor'})
 return a.join(profiles,on=['hand_id','last_aggressor','street_no'],how='left',maintain_order='left').with_columns(*[pl.col(c).fill_null(-1).cast(pl.Float32) for c in COLUMNS])
