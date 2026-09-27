import os,sys
os.environ.setdefault('POLARS_MAX_THREADS','4');sys.path.insert(0,'src/policy')
from pathlib import Path
import csv,json,hashlib,time
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from sequence_features import augment
import build_outcome_roles as BOR,build_relationship_evidence as BRE
from session4_evidence_model import load_models,score,COLS,FAMILIES
from session6_priority_model import load_models as load_priority,score as priority_score
from session7_model import load_models as load_event,score as event_score
OUT=Path(os.environ.get('EVAL_CACHE_OUTPUT','artifacts/evidence_session11/eval_cache'));OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('artifacts/candidate_r29/submission.csv')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='0fe7747d01d1c64099b803fb830f995ae1b4ca055dd1333fd260475732d10c18'
from session6_priority import inclusion
base=pl.read_csv(SOURCE);sel=base.filter((pl.col('risk_score')>=float(os.environ.get('EVAL_CACHE_MIN','.05')))&(pl.col('risk_score')<float(os.environ.get('EVAL_CACHE_MAX','1.01')))).select('pair_id',pl.col('predicted_behavior').alias('behavior_family'))
assert set(sel['behavior_family'])<=set(FAMILIES)
ev=pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2');models=load_models()
priority_models,priority_cols=load_priority('priority_ordered');assert priority_cols==COLS
event_models,event_cols=load_event('hist_eventblend');assert event_cols==COLS
evidence_scores=[]
selections={};C=pl.col;root=Path('artifacts/policy');t=time.time();done=0
with threadpool_limits(limits=4):
    for path in sorted(Path('artifacts/detail_features').glob('*.parquet')):
        if (OUT/path.name).exists():continue
        table_parts=[]
        d=pl.read_parquet(path).filter(C('phase')=='evaluation').join(sel,on='pair_id')
        if d.is_empty():continue
        d=d.with_columns(((C('time')-.6)/.4).alias('relative_time'))
        h=pl.read_parquet(root/'hand_features'/path.name).filter(C('phase')=='evaluation').join(sel.select('pair_id'),on='pair_id').sort('pair_id','time_index')
        rc=[c for c in h.columns if c.endswith('_r')]
        h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')]
        h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
        d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']);d,_=augment(d)
        query=d.select('pair_id','hand_id').join(ev,on='pair_id');d=d.join(BOR.build(path.stem,query),on=['pair_id','hand_id']).join(BRE.build(path.stem,query),on=['pair_id','hand_id'])
        assert d.select(COLS).null_count().to_numpy().sum()==0
        for b in FAMILIES:
            q=d.filter(C('behavior_family')==b)
            if q.is_empty():continue
            x=q.select(COLS).to_numpy();assert np.isfinite(x).all()
            q=q.sort('pair_id','time','hand_id');x=q.select(COLS).to_numpy();raw={}
            for f in range(4):
                bp=score(models,b,x,[f]);ca=priority_models[b][f][0].predict_proba(x,thread_count=4)[:,1];cb=priority_models[b][f][1].predict_proba(x,thread_count=4)[:,1];heads=event_models[b][f][1];ha=heads[0].predict_proba(x)[:,1];hb=heads[1].predict_proba(x)[:,1]
                raw.update({f'{name}_{f}':v for name,v in [('base',bp),('cat_primary',ca),('cat_secondary',cb),('hist_primary',ha),('hist_secondary',hb)]})
            q=q.with_columns(*[pl.Series(k,v) for k,v in raw.items()]);parts=[]
            for _,g in q.group_by('pair_id'):
                g=g.sort('time','hand_id')
                for f in range(4):
                    ca=g[f'cat_primary_{f}'].to_numpy();cb=g[f'cat_secondary_{f}'].to_numpy();ha=g[f'hist_primary_{f}'].to_numpy();hb=g[f'hist_secondary_{f}'].to_numpy();cs=np.maximum(1,ca+cb);hs=np.maximum(1,ha+hb);p=inclusion(ca,cb);j=inclusion(.5*(ca/cs+ha/hs),.5*(cb/cs+hb/hs));g=g.with_columns(pl.Series(f'cat_inclusion_{f}',p),pl.Series(f'joint_inclusion_{f}',j),pl.Series(f'r29_{f}',.25*g[f'base_{f}'].to_numpy()+.25*p+.5*j))
                parts.append(g)
            q=pl.concat(parts);keep=list(dict.fromkeys(['pair_id','hand_id','behavior_family','time','relative_time']+COLS+[f'{n}_{f}' for f in range(4) for n in ['base','cat_primary','cat_secondary','hist_primary','hist_secondary','cat_inclusion','joint_inclusion','r29']]));table_parts.append(q.select(keep))
        pl.concat(table_parts).write_parquet(OUT/path.name)
        done+=1
        if done%30==0:print('candidate tables',done,'seconds',round(time.time()-t,1),flush=True)
cached=pl.read_parquet(list(OUT.glob('T*.parquet')));assert set(cached['pair_id'])==set(sel['pair_id']);assert cached.select('pair_id','hand_id').n_unique()==len(cached)
old=pl.read_parquet('artifacts/candidate_r29/evidence_scores.parquet').select('pair_id','hand_id','base_score');z=cached.with_columns(pl.mean_horizontal([f'r29_{f}' for f in range(4)]).alias('replay')).join(old,on=['pair_id','hand_id'],validate='1:1');err=float((z['replay']-z['base_score']).abs().max()) if len(z) else None;assert err is None or err<1e-10
(OUT/'audit.json').write_text(json.dumps({'hands':len(cached),'pairs':cached['pair_id'].n_unique(),'r29_replay_max_error':err,'replay_overlap_hands':len(z),'min_risk':float(os.environ.get('EVAL_CACHE_MIN','.05')),'max_risk':float(os.environ.get('EVAL_CACHE_MAX','1.01')),'columns':cached.columns},indent=2));print('evaluation cache verified',len(cached),err,flush=True)
