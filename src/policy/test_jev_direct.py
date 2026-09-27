import json,tempfile
from pathlib import Path
import session210_jev_direct as d

def main():
 calls=[];failed=[False]
 with tempfile.TemporaryDirectory(prefix='jev210_test_') as td:
  d.ROOT=Path(td);d.root('pilot').mkdir();d.credential=lambda:'synthetic';d.RATE=10000
  fakejobs=[]
  for k in range(2):
   requests=[d.payload({'schema':d.SCHEMA,'hands':[{'id':f'h{i}','actions':[],'players':[]}],
                        'test_pair':k,'chunk_index':i}) for i in range(3)]
   fakejobs.append({'pair_id':str(k),'mapping':{f'h{i}':f'H{k}{i}' for i in range(3)},'requests':requests})
  d.jobs=lambda scope:iter(fakejobs)
  def fake(key,p):
   calls.append(d.digest(p))
   if p['state'].get('test_pair')==0 and p['state'].get('chunk_index')==1 and not failed[0]:
    failed[0]=True;raise RuntimeError('injected failure')
   answers={}
   for k,q in p['questions'].items():
    if q['type']=='noul':answers[k]={'type':'noul','noul':.4}
    else:
     options=list(q['criteria']);chosen=options[1] if k=='evidence' else options[0]
     answers[k]={'type':'choice','choice':chosen,'confidence':1.,
                 'probabilities':{x:float(x==chosen) for x in options}}
   return {'model':d.MODEL,'answers':answers,'usage':{'input_tokens':10}},0
  d.evaluate=fake
  try:d.infer('pilot')
  except RuntimeError:pass
  else:raise AssertionError('Expected failure')
  assert json.loads((d.root('pilot')/'progress.json').read_text())['status']=='failed'
  d.infer('pilot');assert len(calls)==11                                               
  d.infer('pilot');assert len(calls)==11
  rows=[json.loads(l) for l in (d.root('pilot')/'conclusions.jsonl').open()]
  assert len(rows)==2
  assert all(r['risk_score']==.4 and r['confidence']==1 and len(r['evidence'])==1 for r in rows)
  records=[json.loads(l) for l in (d.root('pilot')/'responses.jsonl').open()]
  assert len(records)==len({(r['pair_id'],r['stage']) for r in records})==10
 print('PASS: full hierarchical hand coverage, direct answers, drained failure, exact resume, no repeated successful calls.')

if __name__=='__main__':main()
