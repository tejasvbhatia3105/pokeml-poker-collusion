import os,polars as pl, numpy as np, json
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
j2=pl.read_parquet(S+'pump_probe2_eval.parquet').select('pair_id','zz','mem','r41','risk_score','n_s','b_n_s','r_s','b_r_s','r_w','b_r_w')
ev=pl.read_parquet(S+'card_share_eval.parquet')
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(pl.max_horizontal(z1,z2).alias('zpos')).select('pair_id','zpos','predicted_behavior')
j=j2.join(ev,on='pair_id')
null=j.filter(C('risk_score')<0.001)
def cnt(x,a,b): return int(((x['zpos']>a)&(x['zz']>b)).sum())
for a,b in [(2.0,2.0),(2.5,2.0),(2.0,2.5),(2.5,2.5),(3.0,2.0)]:
    fr=cnt(null,a,b)/null.height
    print(f'zpos>{a} & zz>{b}: null frac {fr:.5f} | members {cnt(j.filter(C("mem")),a,b)}/53 | r41 {cnt(j.filter(C("r41")),a,b)}/12 |', ' '.join(f'{name}:{cnt(j.filter(f&~C("mem")&~C("r41")),a,b)}(exp {j.filter(f&~C("mem")&~C("r41")).height*fr:.1f})' for name,f in [('0.5-0.9',(C('risk_score')>=0.5)&(C('risk_score')<0.9)),('0.05-0.5',(C('risk_score')>=0.05)&(C('risk_score')<0.5)),('0.01-0.05',(C('risk_score')>=0.01)&(C('risk_score')<0.05)),('0.001-0.01',(C('risk_score')>=0.001)&(C('risk_score')<0.01))]))
x=j.filter(~C('mem')&~C('r41')&(C('risk_score')>=0.01)&(C('risk_score')<0.9)&(C('zpos')>2.0)&(C('zz')>2.0)).sort('risk_score',descending=True)
pl.Config.set_tbl_rows(40); print(x.select('pair_id','risk_score','predicted_behavior','zpos','zz','n_s','b_n_s','r_s','b_r_s','r_w','b_r_w'))
x.write_parquet(S+'lift_candidates2.parquet')
