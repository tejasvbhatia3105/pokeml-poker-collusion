import os,json,csv,hashlib,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from session46_paired_inference import ROOT,load_models,events,score_pair,old_score
C=pl.col;OUT=Path('artifacts/candidate_r31');SOURCE=Path('artifacts/candidate_r30/submission.csv');SHA='d2e6e55063d8be986d9d009c036db4c64c55ffb23a515db468871be80a92195f'
def main(out=OUT,inference_root=ROOT,model_loader=load_models,event_function=events,method='session41 paired decisions, Cat-and-joint prior; mean four folds after seed-averaged count-conditioned selection',local_map=.7857653823178017,score_function=score_pair,extra_inference_columns=None):
 OUT=out;ROOT=inference_root;load_models=model_loader;events=event_function;score_pair=score_function;extra_inference_columns=extra_inference_columns or []
 assert (ROOT/'verification.json').exists();assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA;OUT.mkdir(exist_ok=True);cache=OUT/'inference';cache.mkdir(exist_ok=True)
 with SOURCE.open(newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 lookup={r['pair_id']:r for r in rows};chosen={r['pair_id'] for r in rows if float(r['risk_score'])>=.05 and r['predicted_behavior'] in ['directed_transfer','soft_play','coordinated_isolation']};players=pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2');models=load_models();r30=pl.read_parquet('artifacts/candidate_r30/evidence_scores.parquet').select('pair_id','hand_id',C('score').alias('r30_score'));parts=[];audit=[];seen=set();start=time.time()
 for i,path in enumerate(sorted(Path('artifacts/evidence_session11/eval_cache').glob('T*.parquet'))):
  d=pl.read_parquet(path).filter(C('pair_id').is_in(list(chosen))).with_columns(pl.lit(path.stem).alias('table_id'));pids=set(d['pair_id']);assert not (seen&pids);seen|=pids
  if not len(d):continue
  assert (d['time']>=.6).all();assert all(lookup[pid]['predicted_behavior']==fam for pid,fam in d.select('pair_id','behavior_family').unique().iter_rows());outpath=cache/path.name
  if outpath.exists():z=pl.read_parquet(outpath);assert set(z['pair_id'])==pids
  else:
   d=events(d,players,models);zs=[];err=0
   for (pid,),g in d.group_by('pair_id'):
    g=g.sort('time','hand_id');scores=[score_pair(g,f,models) for f in range(4)];old=np.mean([old_score(g,f,'conditional_family',models['correction']) for f in range(4)],0);ref=g.select('pair_id','hand_id').join(r30,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['r30_score'].to_numpy();err=max(err,float(abs(old-ref).max()));zs.append(g.select('pair_id','hand_id','time','behavior_family',*[f'new_{head}_{f}' for f in range(4) for head in ['primary','secondary']],*extra_inference_columns).with_columns(*[pl.Series(f'score_{f}',v) for f,v in enumerate(scores)],pl.Series('score',np.mean(scores,0)),pl.Series('r30_score',old)))
   assert err<1e-10;z=pl.concat(zs);z.write_parquet(outpath);outpath.with_suffix('.json').write_text(json.dumps({'table':path.stem,'pairs':len(pids),'hands':len(d),'r30_reference_error':err},indent=2))
  audit.append(json.load(open(outpath.with_suffix('.json'))));parts.append(z)
  if i%40==0:print('paired candidate',i,len(seen),round(time.time()-start,1),flush=True)
 assert seen==chosen;pred=pl.concat(parts);assert np.isfinite(pred['score'].to_numpy()).all();choices={pid:g.sort('score','hand_id',descending=[True,False])['hand_id'].to_list()[:5] for (pid,),g in pred.group_by('pair_id')};ecols=[f'evidence_hand_{j}' for j in range(1,6)];changes=0
 for row in rows:
  if row['pair_id'] not in choices:continue
  hands=choices[row['pair_id']];assert len(hands)==5;changes+=any(row[c]!=h for c,h in zip(ecols,hands))
  for c,h in zip(ecols,hands):row[c]=h
 with SOURCE.open(newline='') as f:
  for old,new in zip(csv.DictReader(f),rows):assert all(old[c]==new[c] for c in fields if c not in ecols);assert old==new or old['pair_id'] in chosen
 with (OUT/'submission.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 pred.write_parquet(OUT/'evidence_scores.parquet');report={'status':'unscored; no claim of .94','source':'R30 .92618 user-reported','source_sha256':SHA,'method':method,'local_evidence_MAP':local_map,'local_R30_evidence_MAP':.7748491636798086,'rows':len(rows),'rescored_pairs':len(chosen),'changed_evidence_rows':changes,'risk_and_behavior_strings_unchanged':True,'below005_rows_unchanged':True,'r30_reference_max_error':max(a['r30_reference_error'] for a in audit),'sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest()};(OUT/'build_manifest.json').write_text(json.dumps(report,indent=2));(OUT/'inference_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
