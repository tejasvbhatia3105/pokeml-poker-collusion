import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6'); sys.path.insert(0,'src/policy')
from pathlib import Path
import argparse,json,time,numpy as np,polars as pl
from catboost import CatBoostClassifier
from sequence_features import augment
import build_outcome_roles as BOR, build_relationship_evidence as BRE
C=pl.col;root=Path('artifacts/policy');NAMES=['directed_transfer','soft_play','coordinated_isolation']
ap=argparse.ArgumentParser();ap.add_argument('--pair-predictions',default=str(root/'residual_eval.csv'));ap.add_argument('--base-submission',default='artifacts/v4/submission.csv');ap.add_argument('--output',required=True);ap.add_argument('--min-risk',type=float,default=0.05);ap.add_argument('--limit',type=int,default=10**9);ap.add_argument('--weight',type=float,default=1.0);args=ap.parse_args()
p=pl.read_csv(args.pair_predictions).with_columns((1-C('none')).alias('risk_score'));p=p.with_columns(pl.Series('family',[NAMES[k] for k in p.select(NAMES).to_numpy().argmax(1)]))
sel=p.filter(C('risk_score')>=args.min_risk).select('pair_id','family');print('pairs to rescore',sel.height,flush=True)
ev=pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2')
oldcols=json.loads(Path('artifacts/rank_columns.json').read_text());seqcols=json.loads((root/'sequence/evidence_columns.json').read_text());relcols=json.loads((root/'relationship_evidence/columns.json').read_text())
models={n:[] for n in NAMES}
for n in NAMES:
    for f in range(4):
        m=CatBoostClassifier();m.load_model(str(root/f'relationship_evidence/{n}_fold{f}.cbm'));models[n].append(m)
cache=root/'submission_hands'                                                          
selections=[];t=time.time();done=0
for i,path in enumerate(sorted(Path('artifacts/detail_features').glob('*.parquet'))):
    table=path.stem
    d=pl.read_parquet(path).filter(C('phase')=='evaluation').join(sel,on='pair_id')
    if d.is_empty():continue
    d=d.with_columns(((C('time')-.6)/.4).alias('relative_time'))
    h=pl.read_parquet(root/'hand_features'/path.name).filter(C('phase')=='evaluation').join(sel.select('pair_id'),on='pair_id').sort('pair_id','time_index');rcols=[c for c in h.columns if c.endswith('_r')]
    h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rcols]);hz=[c for c in h.columns if c.endswith('_hz')]
    h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
    d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']);d,_=augment(d)
    query=d.select('pair_id','hand_id').join(ev,on='pair_id')
    ro=BOR.build(table,query);re=BRE.build(table,query)
    d=d.join(ro,on=['pair_id','hand_id']).join(re,on=['pair_id','hand_id'])
    assert d.select(relcols).null_count().to_numpy().sum()==0
    v4=pl.read_parquet(cache/path.name).select('pair_id','hand_id','hand_score')
    scores=[]
    for n in NAMES:
        z=d.filter(C('family')==n)
        if z.is_empty():continue
        s=np.mean([m.predict_proba(z.select(relcols).to_numpy(),thread_count=6)[:,1] for m in models[n]],axis=0)
        scores.append(z.select('pair_id','hand_id','family').with_columns(pl.Series('rel_score',s)))
    scores=pl.concat(scores).join(v4,on=['pair_id','hand_id'],how='left').with_columns((args.weight*C('rel_score')+(1-args.weight)*C('hand_score').fill_null(0)).alias('score'))
    ranked=scores.sort(['pair_id','score','hand_id'],descending=[False,True,False]);selections.append(ranked.group_by('pair_id',maintain_order=True).agg(C('hand_id').head(5).alias('hands')))
    done+=1
    if done%20==0:print(i,'tables',done,'seconds',round(time.time()-t,1),flush=True)
    if done>=args.limit:break
e=pl.concat(selections).with_columns(*[C('hands').list.get(k,null_on_oob=True).fill_null('NO_EVIDENCE').alias(f'evidence_hand_{k+1}') for k in range(5)]).drop('hands')
print('rescored pairs',e.height,'seconds',round(time.time()-t,1),flush=True)
base=pl.read_csv(args.base_submission)
r=base.select('pair_id').join(p.select('pair_id','risk_score',pl.when(C('risk_score')<.01).then(pl.lit('none')).otherwise(C('family')).alias('predicted_behavior')),on='pair_id',how='left')
ecols=[f'evidence_hand_{k}' for k in range(1,6)]
r=r.join(e,on='pair_id',how='left').join(base.select('pair_id',*[C(c).alias(c+'_v4') for c in ecols]),on='pair_id',how='left')
r=r.with_columns(*[pl.coalesce([C(c),C(c+'_v4')]).alias(c) for c in ecols]).select('pair_id','risk_score','predicted_behavior',*ecols)
template=pl.read_csv('data/sample_submission.csv');r=template.select('pair_id').join(r,on='pair_id',how='left',maintain_order='left').select(template.columns)
r.write_csv(args.output);print('saved',args.output,r.shape,flush=True)
