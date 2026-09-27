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
OUT=Path(os.environ.get('R28_OUTPUT','artifacts/candidate_r28'));OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'submission.csv').exists(), 'Preserve existing candidates; use a new output directory'
SOURCE=Path('artifacts/candidate_r27/submission.csv')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='1cef3d932b0fea2e669980ab8b4f367f89352b5aaf1553bb3a86b2546873f9ae'
base=pl.read_csv(SOURCE);sel=base.filter(pl.col('risk_score')>=.05).select('pair_id',pl.col('predicted_behavior').alias('behavior_family'))
assert set(sel['behavior_family'])<=set(FAMILIES)
ev=pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2');models=load_models()
priority_models,priority_cols=load_priority('priority_ordered');assert priority_cols==COLS
evidence_scores=[]
selections={};C=pl.col;root=Path('artifacts/policy');t=time.time();done=0
with threadpool_limits(limits=4):
    for path in sorted(Path('artifacts/detail_features').glob('*.parquet')):
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
            base_probability=score(models,b,x,range(4));priority_probability=priority_score(priority_models,priority_cols,b,q,range(4))
            q=q.with_columns(pl.Series('base_score',.5*base_probability+.5*priority_probability),pl.Series('r27_base_score',base_probability),pl.Series('priority_score',priority_probability))
            evidence_scores.append(q.select('pair_id','hand_id','behavior_family','time','base_score','r27_base_score','priority_score'))
            for (pid,),g in q.group_by('pair_id'):
                assert pid not in selections;selections[pid]=g.sort(['base_score','hand_id'],descending=[True,False])['hand_id'].to_list()[:5]
        done+=1
        if done%30==0:print('candidate tables',done,'seconds',round(time.time()-t,1),flush=True)
assert set(selections)==set(sel['pair_id'])
pl.concat(evidence_scores).write_parquet(OUT/'evidence_scores.parquet')
with SOURCE.open(newline='') as source:
    reader=csv.DictReader(source);fields=reader.fieldnames;rows=list(reader)
ecols=[f'evidence_hand_{i}' for i in range(1,6)];changed=0
for row in rows:
    if row['pair_id'] in selections:
        chosen=selections[row['pair_id']];new=[chosen[i] if i<len(chosen) else 'NO_EVIDENCE' for i in range(5)]
        changed+=any(row[c]!=h for c,h in zip(ecols,new))
        for c,h in zip(ecols,new):row[c]=h
with (OUT/'submission.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
with SOURCE.open(newline='') as file:
    original=list(csv.DictReader(file))
for a,b in zip(original,rows):
    assert all(a[c]==b[c] for c in fields if c not in ecols)
report={'source':'R27','status':'unscored_evidence_only_candidate','rows':len(rows),'rescored_pairs':len(selections),'changed_evidence_rows':changed,
        'non_evidence_fields_preserved_as_exact_strings':True,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'sha256':hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest(),'method':'50/50 R27 base probabilities and ordered-priority capped inclusion probabilities; average inclusion AFTER selection separately within each of four fold models; no unary correction; phase-specific histories; legacy 811 features; min risk .05'}
report['local_oof']=json.loads(Path('artifacts/evidence_session6/comparison.json').read_text())['priority_ordered_half_r27_raw']
report['models']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('artifacts/evidence_session6').glob('priority_ordered_event*.cbm'))}
(OUT/'build_manifest.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
