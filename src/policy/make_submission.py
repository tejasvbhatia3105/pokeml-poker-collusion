import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import argparse,json,time,numpy as np,polars as pl
from catboost import CatBoostClassifier
from sequence_features import augment
C=pl.col;root=Path('artifacts/policy');NAMES=['directed_transfer','soft_play','coordinated_isolation']
parser=argparse.ArgumentParser();parser.add_argument('--pair-predictions',default=str(root/'residual_eval.csv'));parser.add_argument('--output',default='submission_v4.csv');args=parser.parse_args()
p=pl.read_csv(args.pair_predictions).with_columns((1-C('none')).alias('risk_score'));p=p.with_columns(pl.Series('family',[NAMES[k] for k in p.select(NAMES).to_numpy().argmax(1)]))
oldcols=json.loads(Path('artifacts/rank_columns.json').read_text());newcols=json.loads((root/'sequence/evidence_columns.json').read_text());models={}
for name in NAMES:
 models[name]={}
 for kind in ['old','sequence']:
  models[name][kind]=[]
  for f in range(4):
   path=Path(f'artifacts/rank_{name}_fold{f}.cbm') if kind=='old' else root/f'sequence/evidence_{name}_fold{f}.cbm'
   m=CatBoostClassifier();m.load_model(str(path));models[name][kind].append(m)
selections=[];t=time.time();cache=root/'submission_hands';cache.mkdir(exist_ok=True)
for i,path in enumerate(sorted(Path('artifacts/detail_features').glob('*.parquet'))):
 d=pl.read_parquet(path).filter(C('phase')=='evaluation').join(p.select('pair_id','family'),on='pair_id').with_columns(((C('time')-.6)/.4).alias('relative_time'))
 h=pl.read_parquet(root/'hand_features'/path.name).filter(C('phase')=='evaluation').sort('pair_id','time_index');rcols=[c for c in h.columns if c.endswith('_r')]
 h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rcols]);hz=[c for c in h.columns if c.endswith('_hz')]
 h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
 d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']);d,_=augment(d);scores=[]
 for name in NAMES:
  z=d.filter(C('family')==name)
  if not len(z):continue
  op=np.mean([m.predict_proba(z.select(oldcols).to_numpy(),thread_count=4)[:,1] for m in models[name]['old']],axis=0)
  npred=np.mean([m.predict_proba(z.select(newcols).to_numpy(),thread_count=4)[:,1] for m in models[name]['sequence']],axis=0)
  scores.append(z.select('pair_id','hand_id','family').with_columns(pl.Series('hand_score',.25*op+.75*npred)))
 scores=pl.concat(scores);scores.write_parquet(cache/path.name,compression='zstd')
 ranked=scores.sort(['pair_id','hand_score','hand_id'],descending=[False,True,False]);selections.append(ranked.group_by('pair_id',maintain_order=True).agg(C('hand_id').head(5).alias('hands')))
 if i%20==0:print(i,'seconds',round(time.time()-t,1),flush=True)
e=pl.concat(selections).with_columns(*[C('hands').list.get(i,null_on_oob=True).fill_null('NO_EVIDENCE').alias(f'evidence_hand_{i+1}') for i in range(5)]).drop('hands')
r=p.select('pair_id','risk_score',pl.when(C('risk_score')<.01).then(pl.lit('none')).otherwise(C('family')).alias('predicted_behavior')).join(e,on='pair_id',how='left').fill_null('NO_EVIDENCE');template=pl.read_csv('data/sample_submission.csv');r=template.select('pair_id').join(r,on='pair_id',how='left',maintain_order='left').select(template.columns);r.write_csv(args.output);print('saved',args.output,r.shape,flush=True)
