\
\
\
import os,sys,glob,argparse,json,time; os.environ.setdefault('POLARS_MAX_THREADS','6')
import numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
C=pl.col; root=Path('artifacts/policy')
ap=argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('phase'); ap.add_argument('rows'); ap.add_argument('out'); ap.add_argument('--min-risk',type=float,default=0.02); ap.add_argument('--base'); ap.add_argument('--policy',default='min'); ap.add_argument('--save-features'); ap.add_argument('--mode',default='top',choices=['top','all','sym']); ap.add_argument('--all-thr',type=float,default=0.3); ap.add_argument('--window')
a=ap.parse_args()
cols=json.loads((root/'residual_columns.json').read_text()); names=['none','directed_transfer','soft_play','coordinated_isolation']
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
models={f:[] for f in range(4)}
for f in range(4):
    for p in sorted(glob.glob(f'{a.model}/pair_seed*_fold{f}.cbm')) or [f'{a.model}/pair_fold{f}.cbm']:
        m=CatBoostClassifier(); m.load_model(p); models[f].append(m)
def predict(X,table):
    if a.phase=='development': ms=models[tf[table]]
    else: ms=[m for f in range(4) for m in models[f]]
    return np.mean([m.predict_proba(X,thread_count=6) for m in ms],axis=0)
channels=['alive_agg','alive_fold','partner_call','weak_partner_call','partner_surrender','hu_passivity','hu_check','outsider_agg','weak_outsider_agg','dealt_weak_agg','hidden_agg_alive','hidden_fold_alive','hidden_agg_folded','hidden_call_folded','yield_better','size_partner','size_outsider']
def aggregate(h):
    keys=['pair_id','phase','player_1','player_2']; h=h.sort('pair_id','phase','time_index')
    stats=[pl.len().alias('policy_n_hands'),C('surprise_max').mean().alias('policy_surprise_mean'),C('surprise_max').top_k(5).mean().alias('policy_surprise_top5')]
    for n in channels:
        r=C(n+'_r');v=C(n+'_v');hz=r/(v+1).sqrt()
        stats+=[(r.sum()/(v.sum()+1).sqrt()).alias(n+'_z'),hz.mean().alias(n+'_mean'),hz.std().alias(n+'_std'),hz.top_k(5).mean().alias(n+'_top5'),(-hz).top_k(5).mean().alias(n+'_bottom5')]
    f=h.group_by(keys).agg(stats)
    roll=[(C(n+'_r').rolling_sum(w,min_samples=1).over(['pair_id','phase'])/(C(n+'_v').rolling_sum(w,min_samples=1).over(['pair_id','phase'])+1).sqrt()).alias(n+f'_w{w}') for w in [5,10,20] for n in channels]
    rh=h.with_columns(roll)
    f=f.join(rh.group_by(keys).agg(*[e for w in [5,10,20] for n in channels for e in [C(n+f'_w{w}').max().alias(n+f'_w{w}_max'),C(n+f'_w{w}').min().alias(n+f'_w{w}_min')]]),on=keys)
    return f.with_columns(pl.selectors.float().cast(pl.Float32)).fill_null(0)
                                                
t0=time.time()
if a.base: base=pl.read_csv(a.base).with_columns((1-C('none')).alias('risk')).select('pair_id','risk',*names)
else:
    parts=[]
    PF=(root/'full_window_stress'/a.window/'pair_features') if a.window else (root/'pair_features')
    for p in sorted(PF.glob('*.parquet')):
        if p.stem not in tf and a.phase=='development': continue
        z=pl.read_parquet(p).filter(C('phase')==a.phase); pr=predict(z.select(cols).to_numpy(),p.stem)
        parts.append(z.select('pair_id').with_columns(*[pl.Series(n,pr[:,k]) for k,n in enumerate(names)]))
    base=pl.concat(parts).with_columns((1-C('none')).alias('risk'))
    print('base predicted',base.height,round(time.time()-t0),flush=True)
PF=(root/'full_window_stress'/a.window/'pair_features') if a.window else (root/'pair_features')
pf=pl.concat([pl.read_parquet(p).filter(C('phase')==a.phase).select('pair_id','player_1','player_2','table_id') for p in sorted(PF.glob('*.parquet'))])
WB={'first_2000':(0,2000),'last_2000':(1000,3000)}.get(a.window) if a.window else None
if WB is None and a.window: WB=tuple(map(int,a.window[1:].split('_')))
base=base.join(pf,on='pair_id')
                           
long=pl.concat([base.select(C('player_1').alias('p'),C('player_2').alias('q'),'risk'),base.select(C('player_2').alias('p'),C('player_1').alias('q'),'risk')])
top=long.sort('risk',descending=True).group_by('p').agg(C('q').first().alias('q1'),C('risk').first().alias('r1'))
b=base.join(top.rename({'p':'player_1','q1':'q1_1','r1':'r1_1'}),on='player_1').join(top.rename({'p':'player_2','q1':'q1_2','r1':'r1_2'}),on='player_2')
b=b.with_columns(((C('q1_1')!=C('player_2'))&(C('risk')<C('r1_1'))).alias('ex1'),((C('q1_2')!=C('player_1'))&(C('risk')<C('r1_2'))).alias('ex2'))
if a.mode=='sym':
    strong=long.filter(C('risk')>a.all_thr).select('p','q',C('risk').alias('rq'))
    b=b.with_columns((C('r1_1')>a.all_thr).alias('ex1'),(C('r1_2')>a.all_thr).alias('ex2'))
if a.mode=='all':
                                                                                                   
    strong=long.filter(C('risk')>a.all_thr).select('p','q',C('risk').alias('rq'))
    b=b.with_columns((C('r1_1')>pl.max_horizontal(pl.lit(a.all_thr),C('risk'))).alias('ex1'),(C('r1_2')>pl.max_horizontal(pl.lit(a.all_thr),C('risk'))).alias('ex2'))
cand=b.filter((C('risk')>=a.min_risk)&(C('ex1')|C('ex2'))); print('candidates',cand.height,'of',b.height,flush=True)
                        
rows=[]; feats=[]
for (table,),q in cand.group_by('table_id'):
    h=pl.read_parquet(Path(a.rows)/f'{table}.parquet').filter(C('phase')==a.phase)
    if WB: h=h.filter((C('time_index')>=WB[0])&(C('time_index')<WB[1]))
    h=h.join(q.select('pair_id','q1_1','q1_2','ex1','ex2'),on='pair_id')
    roster=pl.read_parquet(root/'states'/f'{table}.parquet').select('hand_id','player_id').unique()
                                        
    ex=h.select('pair_id','hand_id','q1_1','q1_2','ex1','ex2')
    if a.mode in ('all','sym'):
        qq=q.select('pair_id','player_1','player_2','risk')
        cond=(C('rq')>pl.max_horizontal(pl.lit(a.all_thr),C('risk'))) if a.mode=='all' else (C('rq')>a.all_thr)
        s1=qq.join(strong.rename({'p':'player_1','q':'qx'}),on='player_1').filter(cond).filter(C('qx')!=C('player_2')).select('pair_id','qx')
        s2=qq.join(strong.rename({'p':'player_2','q':'qx'}),on='player_2').filter(cond).filter(C('qx')!=C('player_1')).select('pair_id','qx')
        ex_all=pl.concat([s1,s2]).unique()
        e1=h.select('pair_id','hand_id').join(ex_all,on='pair_id').join(roster.rename({'player_id':'qx'}),on=['hand_id','qx']).select('pair_id','hand_id'); e2=e1.head(0)
    else:
        e1=ex.filter(C('ex1')).join(roster.rename({'player_id':'q1_1'}),on=['hand_id','q1_1']).select('pair_id','hand_id')
        e2=ex.filter(C('ex2')).join(roster.rename({'player_id':'q1_2'}),on=['hand_id','q1_2']).select('pair_id','hand_id')
    drop=pl.concat([e1,e2]).unique()
    kept=h.join(drop,on=['pair_id','hand_id'],how='anti').drop('q1_1','q1_2','ex1','ex2')
    nk=kept.group_by('pair_id').len().rename({'len':'n_kept'})
    if kept.is_empty(): continue
    f=aggregate(kept); pr=predict(f.select(cols).to_numpy(),table)
    if a.save_features: feats.append(f.select('pair_id',*cols).join(nk,on='pair_id'))
    rows.append(f.select('pair_id').with_columns(*[pl.Series(n+'_new',pr[:,k]) for k,n in enumerate(names)]).join(nk,on='pair_id'))
if a.save_features: pl.concat(feats).write_parquet(a.save_features)
new=pl.concat(rows).with_columns((1-C('none_new')).alias('risk_new')); print('rescored',new.height,round(time.time()-t0),flush=True)
out=base.join(new,on='pair_id',how='left').with_columns(C('n_kept').fill_null(0))
                                                                                   
out=out.with_columns(pl.when(C('risk_new').is_null()).then(C('risk')).when(C('n_kept')<10).then(pl.lit(0.0)).otherwise(C('risk_new')).alias('risk_new'))
if a.policy=='min': adj=pl.min_horizontal('risk','risk_new')
elif a.policy=='new': adj=C('risk_new')
else: adj=(C('risk')*C('risk_new')).sqrt()
out=out.with_columns(adj.alias('risk_adj'))
fam=out.select(names[1:]).to_numpy(); fam=fam/np.maximum(fam.sum(1,keepdims=True),1e-12); r=out['risk_adj'].to_numpy()
res=pl.DataFrame({'pair_id':out['pair_id'],'none':1-r,**{n:r*fam[:,k] for k,n in enumerate(names[1:])},'risk_orig':out['risk'],'risk_new':out['risk_new'],'n_kept':out['n_kept']}).sort('pair_id')
res.write_csv(a.out); print('saved',a.out,'changed',int((np.abs(r-out['risk'].to_numpy())>1e-9).sum()),'>0.5 before',int((out['risk']>0.5).sum()),'after',int((r>0.5).sum()),flush=True)
