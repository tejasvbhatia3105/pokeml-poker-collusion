\
\
\
\
\
\
import os,json,ast,hashlib,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
from window_actions import window_actions
C=pl.col;P=Path('artifacts/policy');ROOT=Path('artifacts/pair_session74_complete_rescore')
CHANNELS=['alive_agg','alive_fold','partner_call','weak_partner_call','partner_surrender','hu_passivity','hu_check','outsider_agg','weak_outsider_agg','dealt_weak_agg','hidden_agg_alive','hidden_fold_alive','hidden_agg_folded','hidden_call_folded','yield_better','size_partner','size_outsider']
def shared_code():
                                                                         
                                                              
 path=Path('src/policy/build_pair_features.py');source=path.read_text();tree=ast.parse(source);loop=next(n for n in tree.body if isinstance(n,ast.For) and 'enumerate' in ast.unparse(n.iter));start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='a');end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='h' and 'base.join' in ast.unparse(n.value));raw=compile(ast.Module(body=loop.body[start:end+1],type_ignores=[]),str(path),'exec')
 other=Path('src/policy/leave_partner_out.py').read_text();fun=next(n for n in ast.parse(other).body if isinstance(n,ast.FunctionDef) and n.name=='aggregate');ns={'pl':pl,'np':np,'C':C,'channels':CHANNELS};exec(compile(ast.Module(body=[fun],type_ignores=[]),'leave_partner_out.aggregate','exec'),ns);return raw,ns['aggregate'],hashlib.sha256(source.encode()).hexdigest()
def models():
 acts=[];sizes=[];pairs={}
 for f in range(4):
  m=CatBoostClassifier();m.load_model(str(P/f'action_fold{f}.cbm'));acts.append(m);m=CatBoostRegressor();m.load_model(str(P/f'size_fold{f}.cbm'));sizes.append(m)
  for v in ['r4s','r4k']:
   pairs[v,f]=[]
   for p in sorted(Path(f'artifacts/candidate_{v}').glob(f'pair_seed*_fold{f}.cbm')):
    m=CatBoostClassifier();m.load_model(str(p));pairs[v,f].append(m)
 return acts,sizes,pairs
def exclusions(q,v):
 z=q.select('pair_id','player_1','player_2','table_id',C(v+'_orig').alias('risk'));long=pl.concat([z.select(C('player_1').alias('p'),C('player_2').alias('qx'),C('risk').alias('rq')),z.select(C('player_2').alias('p'),C('player_1').alias('qx'),C('risk').alias('rq'))]).filter(C('rq')>.3);cand=z.filter(C('risk')>=.02);ex=[]
 for own,partner in [('player_1','player_2'),('player_2','player_1')]:ex.append(cand.join(long.rename({'p':own}),on=own).filter((C('rq')>pl.max_horizontal(C('risk'),pl.lit(.3)))&(C('qx')!=C(partner))).select('pair_id','qx'))
 return pl.concat(ex).unique()
def main():
 ROOT.mkdir(exist_ok=True);raw,aggregate,sourcehash=shared_code();acts,sizes,pmodels=models();folds=json.load(open(P/'table_folds.json'));cols=json.load(open(P/'feature_columns.json'));pcols=json.load(open(P/'residual_columns.json'));known=pl.concat([pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2')]);meta=pl.concat([pl.read_parquet(p,columns=['pair_id','table_id','phase']).filter(C('phase')=='development').drop('phase') for p in sorted((P/'pair_features').glob('*.parquet'))]);audit=json.loads((ROOT/'audit.json').read_text())['runs'] if (ROOT/'audit.json').exists() else [];starttime=time.time()
 for w in os.environ.get('RESCORE_WINDOWS','full,first_2000,last_2000').split(','):
  q=pl.read_parquet(f'artifacts/pair_session73_exclusion_audit/{w}.parquet').join(meta,on='pair_id',validate='1:1');ex={v:exclusions(q,v) for v in ['r4s','r4k']};query=pl.concat([z.select('pair_id') for z in ex.values()]).unique();tables=q.join(query,on='pair_id',how='semi').partition_by('table_id',as_dict=True);parts={v:[] for v in ex};coverage=[];baseerror=0
  for ti,((table,),local) in enumerate(sorted(tables.items())):
   ns={'pl':pl,'np':np,'C':C,'root':P,'path':P/'actions'/f'{table}.parquet','window':None if w=='full' else w,'window_actions':window_actions,'cols':cols,'folds':folds,'table':table,'models':acts,'sizes':sizes,'known':known,'HANDS_ALL':None};exec(raw,ns);h=ns['h'].filter(C('phase')=='development').join(local.select('pair_id'),on='pair_id',how='semi').with_columns(pl.selectors.float().cast(pl.Float32));del ns
                                                                           
   pf=P/'pair_features' if w=='full' else P/'full_window_stress'/w/'pair_features';ref=pl.read_parquet(pf/f'{table}.parquet').select('pair_id','phase',*pcols);f=aggregate(h);joined=f.join(ref,on=['pair_id','phase'],suffix='_ref',validate='1:1');err=float(np.max(abs(joined.select(pcols).to_numpy()-joined.select([c+'_ref' for c in pcols]).to_numpy()),initial=0));baseerror=max(baseerror,err);assert err<2e-4,(table,w,err)
   roster=pl.read_parquet(P/'states'/f'{table}.parquet',columns=['hand_id','player_id']).unique().rename({'player_id':'qx'})
   for v in ex:
    e=ex[v].join(local.select('pair_id'),on='pair_id',how='semi');ids=e.select('pair_id').unique()
    if not len(ids):continue
    hh=h.join(ids,on='pair_id',how='semi');assert hh['pair_id'].n_unique()==len(ids);drop=hh.select('pair_id','hand_id').join(e,on='pair_id').join(roster,on=['hand_id','qx']).select('pair_id','hand_id').unique();kept=hh.join(drop,on=['pair_id','hand_id'],how='anti');cnt=kept.group_by('pair_id').len().rename({'len':'n_kept'});f=aggregate(kept);prob=np.mean([m.predict_proba(f.select(pcols).to_numpy(),thread_count=2) for m in pmodels[v,folds[table]]],0) if len(f) else np.zeros((0,4));new=ids.join(f.select('pair_id').with_columns(pl.Series('raw_new',1-prob[:,0])),on='pair_id',how='left',validate='1:1').join(cnt,on='pair_id',how='left',validate='1:1').with_columns(C('n_kept').fill_null(0)).with_columns(pl.when(C('n_kept')<10).then(0).otherwise(C('raw_new')).alias('risk_new'));parts[v].append(new);coverage.append({'window':w,'model':v,'table_id':table,'candidates':len(ids),'with_hand_rows':hh['pair_id'].n_unique(),'zero_kept':new.filter(C('n_kept')==0).height})
   if ti%25==0:print(w,ti,len(tables),'raw_error',baseerror,'seconds',round(time.time()-starttime,1),flush=True)
  for v in ex:
   new=pl.concat(parts[v]);assert len(new)==ex[v]['pair_id'].n_unique();old=pl.read_csv(f'artifacts/evidence_session9/{v}_{w}.csv');out=old.rename({'risk_new':'archived_new','n_kept':'archived_kept','none':'archived_none'}).join(new,on='pair_id',how='left',validate='1:1');risk=out.select(pl.min_horizontal(C('risk_orig'),pl.coalesce('risk_new','risk_orig'))).to_series().to_numpy();fam=out.select('directed_transfer','soft_play','coordinated_isolation').to_numpy();fam/=np.maximum(fam.sum(1,keepdims=True),1e-300);out=out.with_columns(pl.Series('none',1-risk),*[pl.Series(c,risk*fam[:,i]) for i,c in enumerate(['directed_transfer','soft_play','coordinated_isolation'])],pl.coalesce('risk_new','risk_orig').alias('risk_new'),C('n_kept').fill_null(0));out.write_parquet(ROOT/f'{v}_{w}.parquet');audit.append({'window':w,'model':v,'qualifying_candidates':len(new),'all_candidates_have_raw_hands':True,'raw_feature_max_error':baseerror,'changed_vs_archive':int((abs(out['none']-out['archived_none'])>1e-12).sum()),'coverage':coverage})
  (ROOT/'audit.json').write_text(json.dumps({'method':__doc__,'raw_source_sha256':sourcehash,'runs':audit},indent=2))
if __name__=='__main__':main()
