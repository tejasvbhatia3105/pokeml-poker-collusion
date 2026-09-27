\
\
\
\
import os,json,time,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from sequence_features import augment
import build_outcome_roles as BOR,build_relationship_evidence as BRE
from session8_data import hand_data,targets
from session6_priority_model import load_models as cats
from session7_model import load_models as events
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session23_more_pseudo');OLD=Path('artifacts/evidence_session12/unlabelled_pseudo');GATE=Path('artifacts/evidence_session12/pseudo_pair_gate');C=pl.col;FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
CONFIG={'change':'add previously unused odd-hash evaluation registry half','selection':'same pair risk >=.9, teacher family agreement, >=2 hands with Cat/HGB event confidence>.9; low pseudo labels require both<.02','isolation':'outer f excluded from pair/event supervision AND from pseudo gameplay tables','weight_mass':'.5 times original eligible-hand mass, unchanged despite extra data','targets':'soft averaged probabilities, never certified ground truth','cache':'existing core/tail gameplay features when present, exact legacy rebuild otherwise; no cached event scores used for selection'}
def gates():
 reg=pl.read_csv('data/evaluation_pairs.csv');reg=reg.filter(pl.Series([hashlib.sha256(p.encode()).digest()[0]%2==1 for p in reg['pair_id']]));folds=json.load(open('artifacts/policy/table_folds.json'));cols=json.load(open('artifacts/policy/residual_columns.json'));models=[]
 for f in range(4):
  m=CatBoostClassifier();m.load_model(str(GATE/f'pair_fold{f}.cbm'));models.append(m)
 parts=[[] for _ in range(4)]
 for path in sorted(Path('artifacts/policy/pair_features').glob('T*.parquet')):
  d=pl.read_parquet(path).filter(C('phase')=='evaluation').join(reg.select('pair_id'),on='pair_id',validate='1:1');x=d.select(cols).to_numpy();fold=folds[path.stem]
  for f in range(4):
   if f==fold:continue
   p=models[f].predict_proba(x,thread_count=2);keep=1-p[:,0]>=.9;fam=np.array(FAMILIES)[p[:,1:].argmax(1)];parts[f].append(d.select('pair_id','player_1','player_2').with_columns(pl.lit(path.stem).alias('table_id'),pl.lit(fold).alias('pool_fold'),pl.lit(f).alias('teacher_fold'),pl.Series('pair_risk',1-p[:,0]),pl.Series('pair_family',fam)).filter(pl.Series(keep)))
 for f in range(4):pl.concat(parts[f]).write_parquet(ROOT/f'odd_gates_fold{f}.parquet')
 return reg
def features(table,players,cols):
 pieces=[]
 for name in ['eval_cache','tail_cache']:
  path=Path('artifacts/evidence_session11')/name/f'{table}.parquet'
  if path.exists():pieces.append(pl.read_parquet(path).join(players.select('pair_id'),on='pair_id',how='semi').select('pair_id','hand_id','time',*cols))
 cached=pl.concat(pieces) if pieces else None;seen=set(cached['pair_id']) if cached is not None else set();missing=players.filter(~C('pair_id').is_in(list(seen)))
 if not len(missing):return cached,'existing_core_tail'
 path=Path('artifacts/detail_features')/f'{table}.parquet';d=pl.read_parquet(path).filter(C('phase')=='evaluation').join(missing.select('pair_id'),on='pair_id').with_columns(((C('time')-.6)/.4).alias('relative_time'));h=pl.read_parquet(Path('artifacts/policy/hand_features')/path.name).filter(C('phase')=='evaluation').join(missing.select('pair_id'),on='pair_id').sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')];h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')];h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);added=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']];d=d.join(h.select('pair_id','hand_id',*added),on=['pair_id','hand_id'],validate='1:1');d,_=augment(d);q=d.select('pair_id','hand_id').join(missing,on='pair_id');d=d.join(BOR.build(table,q),on=['pair_id','hand_id']).join(BRE.build(table,q),on=['pair_id','hand_id']).select('pair_id','hand_id','time',*cols);assert set(d['pair_id'])==set(missing['pair_id']);return pl.concat(([cached] if cached is not None else [])+[d],how='vertical_relaxed'),'legacy_rebuild_missing'
def cache():
 cm,cols=cats('priority_ordered');hm,_=events('hist_eventblend');reg=gates();allg=[pl.read_parquet(ROOT/f'odd_gates_fold{f}.parquet') for f in range(4)];un=pl.concat(allg).select('pair_id','player_1','player_2','table_id').unique();parts=[[] for _ in range(4)];audit=[];start=time.time()
 with threadpool_limits(limits=2):
  for i,((table,),g) in enumerate(un.group_by('table_id')):
   d,source=features(table,g.drop('table_id'),cols);d=d.sort('pair_id','time','hand_id');x=d.select(cols).to_numpy();assert np.isfinite(x).all();groups=[];off=0
   for (pid,),q in d.group_by('pair_id',maintain_order=True):groups.append((pid,off,off+len(q)));off+=len(q)
   for f in range(4):
    gg=allg[f].filter(C('table_id')==table)
    if not len(gg):continue
    assert (gg['pool_fold']!=f).all();allowed=dict(gg.select('pair_id','pair_family').iter_rows());cp=np.stack([np.column_stack([cm[b][f][k].predict_proba(x,thread_count=2)[:,1] for k in range(2)]) for b in FAMILIES],1);hp=np.stack([np.column_stack([hm[b][f][1][k].predict_proba(x)[:,1] for k in range(2)]) for b in FAMILIES],1);high=(cp>.9)&(hp>.9);low=(cp<.02)&(hp<.02);avg=.5*(cp+hp);selected=0
    for pid,lo,hi in groups:
     if pid not in allowed:continue
     count=high[lo:hi].any(2).sum(0);mass=avg[lo:hi].sum((0,2));best=max(range(3),key=lambda j:(count[j],mass[j]))
     if count[best]<2 or FAMILIES[best]!=allowed[pid]:continue
     eligible=high[lo:hi,best]|low[lo:hi,best];keep=eligible.any(1);ix=np.arange(lo,hi)[keep];pr=avg[ix,best];e=eligible[keep];out=d[ix].select('pair_id','hand_id',*[C(c).cast(pl.Float32) for c in cols]).with_columns(pl.lit(FAMILIES[best]).alias('behavior_family'),pl.lit(int(gg['pool_fold'][0])).alias('pool_fold'),pl.lit(f).alias('teacher_fold'),pl.Series('pseudo_primary',pr[:,0]),pl.Series('pseudo_secondary',pr[:,1]),pl.Series('eligible_primary',e[:,0]),pl.Series('eligible_secondary',e[:,1]));parts[f].append(out);selected+=1
    audit.append({'table':table,'fold':f,'pool_fold':int(gg['pool_fold'][0]),'pair_gated_candidates':len(allowed),'retained_odd_pairs':selected,'feature_source':source})
   if i%30==0:print('extra pseudo',i,round(time.time()-start,1),flush=True)
 for f in range(4):
  odd=pl.concat(parts[f]);assert all(hashlib.sha256(p.encode()).digest()[0]%2==1 for p in odd['pair_id'].unique());odd.write_parquet(ROOT/f'odd_outer{f}.parquet',compression='zstd')
 (ROOT/'cache_audit.json').write_text(json.dumps({'odd_registry_pairs':len(reg),'tables':audit},indent=2))
def train():
 d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];X=d.select(cols).to_numpy();pred=np.zeros((len(d),2));audit=[];start=time.time()
 for f in range(4):
  even=pl.read_parquet(list((OLD/f'outer{f}').glob('T*.parquet'))).join(pl.read_parquet(GATE/f'gate_fold{f}.parquet').filter(C('keep')).select('pair_id'),on='pair_id',validate='m:1');odd=pl.read_parquet(ROOT/f'odd_outer{f}.parquet');assert not set(even['pair_id'])&set(odd['pair_id']);q=pl.concat([even,odd],how='vertical_relaxed');assert (q['pool_fold']!=f).all() and (q['teacher_fold']==f).all();assert q.select('pair_id','hand_id').n_unique()==len(q)
  for fam in FAMILIES:
   z=q.filter(C('behavior_family')==fam);ZX=z.select(cols).to_numpy();a,b,ea,eb,va=targets(d,f,fam)
   for k,(y,e) in enumerate([(a,ea),(b,eb)],1):
    col='primary' if k==1 else 'secondary';use=z['eligible_'+col].to_numpy();px=ZX[use];pr=z['pseudo_'+col].to_numpy()[use];ix=np.flatnonzero(e);assert not np.any(e&va);wgt=.5*len(ix)/max(1,len(px));tx=np.concatenate([X[ix],px,px]);ty=np.r_[y[ix].astype(int),np.zeros(len(px),int),np.ones(len(px),int)];w=np.r_[np.ones(len(ix)),wgt*(1-pr),wgt*pr];m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);path=ROOT/f'event{k}_{fam}_fold{f}.cbm'
    if path.exists():m.load_model(str(path))
    else:m.fit(tx,ty,sample_weight=w);m.save_model(str(path))
    pred[va,k-1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'family':fam,'head':k,'original_rows':len(ix),'pseudo_rows':len(px),'soft_positive_mass':float(pr.sum()),'pseudo_pair_count':z['pair_id'].n_unique(),'even_pairs_all_families':even['pair_id'].n_unique(),'odd_pairs_all_families':odd['pair_id'].n_unique(),'validation_overlap':int((e&va).sum())})
   print('more pseudo student',f,fam,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(pl.Series('bg_primary',pred[:,0]),pl.Series('bg_secondary',pred[:,1])).write_parquet(ROOT/'event_oof.parquet');(ROOT/'training_audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));
 if not all((ROOT/f'odd_outer{f}.parquet').exists() for f in range(4)):cache()
 train()
