\
\
\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','8')
import json,time,numpy as np,polars as pl
from pathlib import Path
C=pl.col; V1=Path(sys.argv[1]); ROWS=Path(sys.argv[2]); OUT=Path(sys.argv[5]); OUT.mkdir(exist_ok=True,parents=True)
channels=['alive_agg','alive_fold','partner_call','weak_partner_call','partner_surrender','hu_passivity','hu_check','outsider_agg','weak_outsider_agg','dealt_weak_agg','hidden_agg_alive','hidden_fold_alive','hidden_agg_folded','hidden_call_folded','yield_better','size_partner','size_outsider']
def risks(path):
    r=pl.read_csv(path).select('pair_id',(1-C('none')).alias('risk'))
    return r
allpairs=pl.concat([pl.read_parquet(p,columns=['pair_id','phase','player_1','player_2']).unique() for p in ROWS.glob('*.parquet')])
R={'development':risks(sys.argv[3]),'evaluation':risks(sys.argv[4])}
names3=[f'q{k}_{n}_z' for k in (1,2) for n in channels]+['q1_risk','q2_risk','q1_dealt','q2_dealt']
json.dump(names3,open(OUT/'columns3.json','w')); t0=time.time()
for i,p in enumerate(sorted(V1.glob('*.npz'))):
    table=p.stem; out=OUT/f'{table}.npy'
    if out.exists(): continue
    z=np.load(p); rows=pl.DataFrame({'row':np.arange(len(z['hand_id'])),'hand_id':z['hand_id'],'pair_id':np.repeat(z['pair_id'],z['n']),'phase':np.repeat(z['phase'],z['n']),'player_1':np.repeat(z['player_1'],z['n']),'player_2':np.repeat(z['player_2'],z['n'])})
    h=pl.read_parquet(ROWS/f'{table}.parquet').with_columns(*[(C(n+'_r')/(C(n+'_v')+1).sqrt()).alias(n+'_z') for n in channels]).select('pair_id','hand_id','phase','player_1','player_2',*[n+'_z' for n in channels])
    parts=[]
    for phase in ['development','evaluation']:
        pr=allpairs.filter((C('phase')==phase)).join(R[phase],on='pair_id',how='left').fill_null(0)
        long=pl.concat([pr.select(C('player_1').alias('p'),C('player_2').alias('q'),'risk'),pr.select(C('player_2').alias('p'),C('player_1').alias('q'),'risk')]).sort('risk',descending=True)
        top2=long.group_by('p').agg(C('q').head(2).alias('qs'),C('risk').head(2).alias('rs'))
        tq={p_:(list(qs),list(rs)) for p_,qs,rs in zip(top2['p'],top2['qs'],top2['rs'])}
        rr=rows.filter(C('phase')==phase)
        if rr.is_empty(): continue
        def other(pl_,x_):
            qs,rs=tq.get(pl_,([],[]))
            for q_,r_ in zip(qs,rs):
                if q_!=x_: return q_,r_
            return None,0.0
        q1=[];r1=[];q2=[];r2=[]
        for a,b in zip(rr['player_1'],rr['player_2']):
            qa,ra=other(a,b); qb,rb=other(b,a); q1.append(qa); r1.append(ra); q2.append(qb); r2.append(rb)
        rr=rr.with_columns(pl.Series('q1',q1),pl.Series('r1',r1),pl.Series('q2',q2),pl.Series('r2',r2))
        hp=h.filter(C('phase')==phase)
                                                                  
        hp=hp.with_columns(pl.min_horizontal('player_1','player_2').alias('lo'),pl.max_horizontal('player_1','player_2').alias('hi'))
        rr=rr.with_columns(pl.min_horizontal('player_1','q1').alias('lo1'),pl.max_horizontal('player_1','q1').alias('hi1'),pl.min_horizontal('player_2','q2').alias('lo2'),pl.max_horizontal('player_2','q2').alias('hi2'))
        j1=rr.join(hp.select('hand_id','lo','hi',*[C(n+'_z').alias(f'q1_{n}_z') for n in channels]).rename({'lo':'lo1','hi':'hi1'}),on=['hand_id','lo1','hi1'],how='left')
        j2=j1.join(hp.select('hand_id','lo','hi',*[C(n+'_z').alias(f'q2_{n}_z') for n in channels]).rename({'lo':'lo2','hi':'hi2'}),on=['hand_id','lo2','hi2'],how='left')
        j2=j2.with_columns(C(f'q1_{channels[0]}_z').is_not_null().cast(pl.Float32).alias('q1_dealt'),C(f'q2_{channels[0]}_z').is_not_null().cast(pl.Float32).alias('q2_dealt'),C('r1').alias('q1_risk'),C('r2').alias('q2_risk')).fill_null(0)
        parts.append(j2.select('row',*names3))
    W=pl.concat(parts).sort('row'); assert W.height==len(z['hand_id']); np.save(out,W.select(names3).to_numpy().astype(np.float16))
    if i%40==0: print(i,table,round(time.time()-t0),flush=True)
print('done',round(time.time()-t0),flush=True)
