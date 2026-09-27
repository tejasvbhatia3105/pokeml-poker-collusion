\
\
\
\
\
\
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path
import session210_jev_direct as core

SOURCE=core.ROOT
ROOT=Path('artifacts/evidence_session211_jev_direct_conditional')
original_questions=core.questions

def questions(ids,final=False):
 q=original_questions(ids,final)
 q['behavior']={
  'type':'choice',
  'instructions':'Conditional question: suppose there is some genuine coordination between A and B somewhere in this supplied history, possibly only in a few hands. Which of the following mechanisms best explains the strongest candidate actions? Classify the mechanism of the suspected episodes, not the majority ordinary hands. This question does not ask whether coordination exists; the separate coordination question assesses that. If uncertain, express uncertainty across these mechanisms.',
  'criteria':{k:v for k,v in core.FAMILIES.items() if k!='none'}}
 return q

def setup(scope):
 r=ROOT/scope;r.mkdir(parents=True,exist_ok=True)
 for name in ['manifest.json','pairs.parquet']+(['labels_private.parquet'] if scope=='pilot' else []):
  dest=r/name
  if not dest.exists():os.link(SOURCE/scope/name,dest)
 if not (r/'hands').exists():(r/'hands').symlink_to((SOURCE/scope/'hands').resolve(),target_is_directory=True)
 protocol={'method':__doc__,'model':core.MODEL,'questions_example':questions(['h0']),
           'source_inputs':str(SOURCE/scope),'notes':'No source response cache reused: questions changed.'}
 path=r/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==protocol
 else:path.write_text(json.dumps(protocol,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','infer','build','run']);ap.add_argument('--scope',choices=['pilot','evaluation'],default='pilot');ap.add_argument('--limit',type=int);a=ap.parse_args()
 setup(a.scope);core.ROOT=ROOT;core.questions=questions
 if a.stage=='preflight':core.preflight(a.scope)
 elif a.stage=='infer':core.infer(a.scope,a.limit)
 else:
  assert a.scope=='evaluation' and a.limit is None,'Submission requires every evaluation pair'
  if a.stage=='run':core.infer('evaluation')
  core.build()
  out=Path('artifacts/candidate_jev_direct')
  subprocess.run([sys.executable,'src/validate_submission.py',str(out/'submission.csv'),
                  '--report',str(out/'validation.json'),'--threads','2','--memory-limit','1GB'],check=True)
  manifest={'model':core.MODEL,'method':__doc__,'source':str(ROOT/'evaluation'),
            'submission_sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest(),
            'leaderboard_score':None,'previous_model_or_submission_used':False}
  (out/'build_manifest.json').write_text(json.dumps(manifest,indent=2))
  (out/'README.md').write_text('# Jev-only submission\n\nAll112,540 pair risks, conditional behavior classes and evidence come directly from Jev1.13.0 applied to raw hand histories. No learned poker model or old submission supplies predictions. Confidence in the sidecar refers to the conditional behavior choice, not coordination probability.\n\nValidated format and raw evidence membership. Unscored on Kaggle. See JEV_DIRECT.md for protocol, costs and pilot limitations.\n')
