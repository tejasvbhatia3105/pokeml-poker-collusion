\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,csv,gzip,hashlib,itertools,json,threading,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import duckdb
import polars as pl
from jev_client import MODEL,PRICE_PER_TOKEN,credential,evaluate

ROOT=Path('artifacts/evidence_session210_jev_direct')
MAX_STATE_BYTES=22000                                                                
RATE=12                                                                      
SCHEMA={
 'game':'Six-player no-limit Texas Holdem. A and B are the candidate pair, not established partners.',
 'units':'All chip amounts in big blinds. Histories chronological; t is timestamp in seconds.',
 'hand_fields':{'id':'Local hand identifier','t':'Time','button':'Button seat number','board':'Full public board; only first3 cards are visible on flop,4 turn,5 river.',
 'players':'Rows [name,seat,hole_cards,starting_stack,contribution,net_result,folded,showdown]. Opponent hole cards omitted. O names refer only to the current seat, not stable identities.',
 'actions':'Chronological rows [street,player,action,incremental_amount,amount_to,pot_before,to_call,stack_before,players_active]. Streets0=preflop,1=flop,2=turn,3=river.'},
 'interpretation':'Hole cards and outcomes are retrospective analyst information. A player legitimately knows only their own cards and the board visible then. Distinguish coordination from weak play, tilt, bluffing, luck, shared style and mere co-occurrence. Coordination can be intermittent.'}
FAMILIES={
 'none':'Ordinary independent play or non-target patterns such as tilt, weakness, coincidence or similar strategy; no supported coordination.',
 'directed_transfer':'A deliberately gives value to B, or B to A, through otherwise unjustified contributions or surrender benefiting the partner.',
 'soft_play':'Partners preferentially avoid normal profitable aggression or conflict against each other.',
 'coordinated_isolation':'Partners coordinate pressure on other players while limiting conflict with each other.',
 'other_coordination':'Positive evidence of another coordinated mechanism, such as actions revealing use of the partner private cards; does not fit the three named mechanisms. Uncertainty alone is not other coordination.'}

def dump(x):return json.dumps(x,separators=(',',':'),ensure_ascii=True)
def digest(x):return hashlib.sha256(dump(x).encode()).hexdigest()
def root(scope):return ROOT/scope

def questions(ids,final=False):
 context='Assess the supplied raw hand history for A and B. If state indicates one chronological chunk, judge that chunk without assuming actions in omitted chunks.' if not final else 'Make the pair judgment for the represented chronological span. The supplied chunk judgments cover disjoint parts of that span; inspect the accompanying original candidate evidence. Chunk answers are fallible model judgments, not ground truth. Do not count the same hand twice.'
 return {
  'coordination':{'type':'noul','instructions':context+' Is there observable evidence that A and B coordinate their poker actions to transfer value, avoid aggression toward each other, isolate others, or otherwise use a joint strategy or private partner information?',
                  'criteria':{'true':'Observed strategic choices support coordination between A and B, possibly episodic.','false':'The history is better explained by independent play, luck, tilt, weakness or common strategy.'}},
  'behavior':{'type':'choice','instructions':context+' Which explanation best describes the relationship between A and B?', 'criteria':FAMILIES},
  'evidence':{'type':'choice','instructions':context+' Which supplied hand contains the strongest observable action evidence of coordination between A and B? Judge the actual choices, not just the chip outcome. The distribution across hands will rank the evidence.',
              'criteria':{'NO_EVIDENCE':'No supplied hand contains observable action evidence of coordination.',**{h:'Hand '+h+' in the supplied history.' for h in ids}}}}

def payload(state,final=False):
 q=questions([h['id'] for h in state['hands']],final)
 p={'model':MODEL,'state':state,'questions':q}
                                                                     
 assert len(dump(state))+max(len(dump(v)) for v in q.values())<32000
 assert len(dump(p))<64000
 assert len(q['evidence']['criteria'])<=255
 return p

def prepare(scope):
 r=root(scope);r.mkdir(parents=True,exist_ok=True)
 assert not (r/'manifest.json').exists(),'Prepared inputs already exist'
 if scope=='pilot':
  labels=pl.read_csv('data/development_labels.csv');selected=[]
  for family,n in [('directed_transfer',4),('soft_play',4),('coordinated_isolation',4),('none',12)]:
   rows=labels.filter(pl.col('behavior_family')==family).to_dicts()
   selected.extend(sorted(rows,key=lambda x:hashlib.sha256(('jev210'+x['pair_id']).encode()).hexdigest())[:n])
  pairs=pl.DataFrame(selected);pairs.write_parquet(r/'labels_private.parquet')
  pairs=pairs.select('pair_id','player_1','player_2');phase='development'
 else:
  pairs=pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2');phase='evaluation'
 pairs.write_parquet(r/'pairs.parquet')
 db=duckdb.connect();db.execute("SET threads=2; SET memory_limit='768MB'; SET temp_directory='/tmp/jev210_duckdb'")
 db.register('requested_pairs',pairs.to_arrow())
 db.execute(f"""CREATE TEMP TABLE keys AS
 SELECT p.pair_id,h.hand_id,h.table_id
 FROM requested_pairs p JOIN read_parquet('data/seats.parquet') s1 ON s1.player_id=p.player_1
 JOIN read_parquet('data/seats.parquet') s2 ON s1.hand_id=s2.hand_id AND s2.player_id=p.player_2
 JOIN read_parquet('data/hands.parquet') h ON h.hand_id=s1.hand_id WHERE h.phase='{phase}'""")
 keycount=db.execute('SELECT count(*) FROM keys').fetchone()[0]
 groups=db.execute('SELECT pair_id, min(table_id) table_id,count(*) shared_hands,count(DISTINCT table_id) nt FROM keys GROUP BY pair_id').pl()
 assert len(groups)==len(pairs) and groups['nt'].max()==1
 pairs=pairs.join(groups.drop('nt'),on='pair_id',validate='1:1');pairs.write_parquet(r/'pairs.parquet')
 if scope=='evaluation':
  truth=pl.read_csv('data/evaluation_pairs.csv').select('pair_id',pl.col('shared_hands').alias('expected'))
  assert pairs.join(truth,on='pair_id').filter(pl.col('shared_hands')!=pl.col('expected')).is_empty()
 db.execute('CREATE TEMP TABLE all_selected_hands AS SELECT DISTINCT hand_id,table_id FROM keys')
 db.execute('DROP TABLE keys')
 db.execute('SET preserve_insertion_order=false')
 query="""WITH ss AS (
 SELECT hand_id,to_json(list(struct_pack(pid:=player_id,seat:=seat_no,c1:=hole_card_1,c2:=hole_card_2,stack:=starting_stack,contribution:=total_contribution,net:=net_chips,fold:=folded,showdown:=went_to_showdown) ORDER BY seat_no)) sj
 FROM read_parquet('data/seats.parquet') SEMI JOIN selected_hands USING(hand_id) GROUP BY hand_id),
 aa AS (SELECT hand_id,to_json(list(struct_pack(street:=street,pid:=player_id,action:=action,amount:=amount,to_amount:=amount_to,pot:=pot_before,to_call_value:=to_call,stack:=stack_before,active:=players_active) ORDER BY action_no)) aj
 FROM read_parquet('data/actions.parquet') SEMI JOIN selected_hands USING(hand_id) GROUP BY hand_id)
 SELECT h.hand_id,h.table_id,epoch(h.started_at),h.button_seat,h.board_cards,h.big_blind,ss.sj,aa.aj
 FROM read_parquet('data/hands.parquet') h JOIN ss USING(hand_id) JOIN aa USING(hand_id)
 ORDER BY h.table_id,h.started_at,h.hand_id"""
 (r/'hands').mkdir(exist_ok=True);nh=0
 tables=sorted(pairs['table_id'].unique().to_list())
 try:
  for offset in range(0,len(tables),20):
   chosen=tables[offset:offset+20]
   db.execute('CREATE OR REPLACE TEMP TABLE selected_hands AS SELECT hand_id FROM all_selected_hands WHERE table_id IN ('+','.join(['?']*len(chosen))+')',chosen)
   cursor=db.execute(query);current=None;out=None
   try:
    while batch:=cursor.fetchmany(256):
     for hand,table,t,button,board,bb,sj,aj in batch:
      if table!=current:
       if out:out.close()
       out=gzip.open(r/'hands'/f'{table}.jsonl.gz','wt',compresslevel=3);current=table
      obj={'hand_id':hand,'t':t,'button':button,'board':board,'bb':bb,'seats':json.loads(sj),'actions':json.loads(aj)}
      out.write(dump(obj)+'\n');nh+=1
   finally:
    if out:out.close()
   print('prepared tables',min(offset+20,len(tables)),'of',len(tables),'hands',nh,flush=True)
 finally:db.close()
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'hands').glob('*.gz')}
 manifest={'scope':scope,'pairs':len(pairs),'unique_hands':nh,'hand_pair_observations':keycount,'model':MODEL,
   'schema':SCHEMA,'family_criteria':FAMILIES,'max_state_bytes':MAX_STATE_BYTES,'cache_sha256':hashes,
   'pairs_sha256':hashlib.sha256((r/'pairs.parquet').read_bytes()).hexdigest(),
   'source':'Raw supplied hands/seats/actions and pair endpoints only. No previous model or submission.',
   'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (r/'manifest.json').write_text(json.dumps(manifest,indent=2));print(dump({k:v for k,v in manifest.items() if k not in ['schema','family_criteria','cache_sha256']}),flush=True)

def hand_state(h,pair,index):
 names={s['pid']:('A' if s['pid']==pair['player_1'] else 'B' if s['pid']==pair['player_2'] else 'O'+str(s['seat'])) for s in h['seats']}
 bb=h['bb'];amount=lambda x:round(x/bb,3)
 return {'id':'h'+str(index),'t':h['t'],'button':h['button'],'board':h['board'],
 'players':[[names[s['pid']],s['seat'],s['c1']+s['c2'] if names[s['pid']] in ('A','B') else '?',amount(s['stack']),amount(s['contribution']),amount(s['net']),int(s['fold']),int(s['showdown'])] for s in h['seats']],
 'actions':[[{'preflop':0,'flop':1,'turn':2,'river':3}[a['street']],names[a['pid']],a['action'],amount(a['amount']),amount(a['to_amount']),amount(a['pot']),amount(a['to_call_value']),amount(a['stack']),a['active']] for a in h['actions']]}

def jobs(scope,pair_ids=None):
 r=root(scope);cfg=json.loads((r/'manifest.json').read_text())
 assert cfg['model']==MODEL and cfg['schema']==SCHEMA and cfg['family_criteria']==FAMILIES
 assert hashlib.sha256((r/'pairs.parquet').read_bytes()).hexdigest()==cfg['pairs_sha256']
 pairs=pl.read_parquet(r/'pairs.parquet')
 if pair_ids is not None:pairs=pairs.filter(pl.col('pair_id').is_in(pair_ids))
 for (table,),g in pairs.sort('table_id','pair_id').group_by('table_id',maintain_order=True):
  path=r/'hands'/f'{table}.jsonl.gz';assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['cache_sha256'][path.name]
  hands=[json.loads(line) for line in gzip.open(path,'rt')];byplayer={}
  for i,h in enumerate(hands):
   for s in h['seats']:byplayer.setdefault(s['pid'],set()).add(i)
  for pair in g.to_dicts():
   indices=sorted(byplayer[pair['player_1']]&byplayer[pair['player_2']]);assert len(indices)==pair['shared_hands']
   states=[hand_state(hands[i],pair,k) for k,i in enumerate(indices)]
   mapping={'h'+str(k):hands[i]['hand_id'] for k,i in enumerate(indices)}
   chunks=[];chunk=[]
   fixed_bytes=len(dump({'schema':SCHEMA,'hands':[],'total_shared_hands':len(states)}));chunk_bytes=fixed_bytes
   for h in states:
    hand_bytes=len(dump(h))
    if chunk and (chunk_bytes+hand_bytes+1>MAX_STATE_BYTES or len(chunk)>=200):
     chunks.append(chunk);chunk=[];chunk_bytes=fixed_bytes
    chunk_bytes+=hand_bytes+(1 if chunk else 0);chunk.append(h)
   if chunk:chunks.append(chunk)
   requests=[payload({'schema':SCHEMA,'hands':ch,'total_shared_hands':len(states),'chunk_index':i,'chunks':len(chunks),'coverage':'All shared hands, divided chronologically without sampling.'}) for i,ch in enumerate(chunks)]
   yield {'pair_id':pair['pair_id'],'mapping':mapping,'requests':requests}

def preflight(scope):
 rows=pairs=chunks=totalbytes=maxbytes=0
 for job in jobs(scope):
  pairs+=1;rows+=len(job['mapping']);chunks+=len(job['requests'])
  for p in job['requests']:
   size=len(dump(p));totalbytes+=size;maxbytes=max(maxbytes,size)
 cfg={'pairs':pairs,'hand_pair_observations':rows,'chunk_calls':chunks,'chunk_payload_bytes':totalbytes,'max_payload_bytes':maxbytes,
      'additional_merge_calls':chunks-pairs,'note':'Token/cost estimate calibrated from pilot responses; all selected raw histories are covered.'}
 (root(scope)/'preflight.json').write_text(json.dumps(cfg,indent=2));print(dump(cfg),flush=True)

def _infer(scope,limit=None):
 r=root(scope);dest=r/'responses.jsonl';cache={};lock=threading.Lock();next_start=[0.];tokens=[0];counts=[0]
 if dest.exists():
  for line in dest.open():
   v=json.loads(line);k=(v['pair_id'],v['stage']);assert k not in cache;cache[k]=v
   tokens[0]+=v['response']['usage']['input_tokens'];counts[0]+=1
 key=credential();start=time.monotonic();failures=[];completed=[0]
 with dest.open('a',buffering=1) as output:
  def ask(pid,stage,p):
   sha=digest(p);k=(pid,stage)
   with lock:
    old=cache.get(k)
    if old:
     assert old['request_sha256']==sha;return old['response']
    now=time.monotonic();delay=max(0,next_start[0]-now);next_start[0]=max(now,next_start[0])+1/RATE
   if delay:time.sleep(delay)
   response,seconds=evaluate(key,p)
   record={'pair_id':pid,'stage':stage,'request_sha256':sha,'response':response,'seconds':seconds}
   with lock:
    output.write(dump(record)+'\n');cache[k]=record;tokens[0]+=response['usage']['input_tokens'];counts[0]+=1
   return response
  def one(job):
   pid=job['pair_id'];answers=[];shortlist=[]
   for i,p in enumerate(job['requests']):
    answer=ask(pid,'chunk'+str(i),p);answers.append(answer['answers'])
    ranked=sorted(answer['answers']['evidence']['probabilities'],key=lambda h:(-answer['answers']['evidence']['probabilities'][h],h))
    ids=[h for h in ranked if h!='NO_EVIDENCE'][:5]
    shortlist.extend(h for h in p['state']['hands'] if h['id'] in ids)
   nodes=[]
   for response,p in zip(answers,job['requests'],strict=True):
    pr=response['evidence']['probabilities']
    ids=[h for h in sorted(pr,key=lambda h:(-pr[h],h)) if h!='NO_EVIDENCE'][:5]
    nodes.append({'covered_hands':len(p['state']['hands']),'answers':response,'hands':[next(h for h in p['state']['hands'] if h['id']==hid) for hid in ids]})
   level=0
   while len(nodes)>1:
    reduced=[]
    for group_index in range(0,len(nodes),2):
     group=nodes[group_index:group_index+2]
     if len(group)==1:reduced.extend(group);continue
     summaries=[{k:dict(v) for k,v in n['answers'].items()} for n in group]
     for a in summaries:a['evidence']={k:v for k,v in a['evidence'].items() if k!='probabilities'}
     selected=[list(n['hands']) for n in group]
     state={'schema':SCHEMA,'hands':sum(selected,[]),'total_shared_hands':len(job['mapping']),
            'represented_hands':sum(n['covered_hands'] for n in group),'chunk_judgments':summaries,'coverage':'These judgments summarize disjoint chronological parts of the shared history. Every shared hand is read at the leaf stage. Original evidence shortlists accompany the judgments.'}
     while len(dump(state))>MAX_STATE_BYTES:
      longest=max(range(len(selected)),key=lambda k:len(dump(selected[k])))
      assert len(selected[longest])>1,'Single evidence hands exceed context budget'
      selected[longest].pop();state['hands']=sum(selected,[])
     response=ask(pid,f'merge{level}_{group_index//2}',payload(state,True))
     pr=response['answers']['evidence']['probabilities']
     ids=[h for h in sorted(pr,key=lambda h:(-pr[h],h)) if h!='NO_EVIDENCE'][:5]
     reduced.append({'covered_hands':sum(n['covered_hands'] for n in group),'answers':response['answers'],'hands':[next(h for h in state['hands'] if h['id']==hid) for hid in ids]})
    nodes=reduced;level+=1
   assert nodes[0]['covered_hands']==len(job['mapping'])
   answer={'model':MODEL,'answers':nodes[0]['answers']}
   a=answer['answers'];probs=a['evidence']['probabilities']
   evidence=[h for h in sorted(probs,key=lambda h:(-probs[h],h)) if h!='NO_EVIDENCE' and probs[h]>0][:5]
   conclusion={'pair_id':pid,'risk_score':a['coordination']['noul'],'predicted_behavior':a['behavior']['choice'],
    'confidence':a['behavior']['confidence'],'class_probabilities':a['behavior']['probabilities'],
    'evidence':[job['mapping'][h] for h in evidence],'chunks':len(answers)}
   with lock:
    conclusions.write(dump(conclusion)+'\n');completed[0]+=1
   return conclusion
                                                                          
  with (r/'conclusions.jsonl').open('w',buffering=1) as conclusions,ThreadPoolExecutor(max_workers=6) as pool:
   source=iter(itertools.islice(jobs(scope),limit));pending=set();exhausted=False;last=0
   while pending or not exhausted:
    while not failures and not exhausted and len(pending)<6:
     job=next(source,None)
     if job is None:exhausted=True;break
     pending.add(pool.submit(one,job))
    if not pending:break
    ready,pending=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
    for future in ready:
     try:future.result()
     except Exception as e:failures.append(type(e).__name__+': '+str(e))
    now=time.monotonic()
    if now-last>15 or not pending:
     status={'completed_pairs':completed[0],'successful_requests':counts[0],'input_tokens':tokens[0],
       'estimated_cost_usd':tokens[0]*PRICE_PER_TOKEN,'seconds_this_run':round(now-start,1),
       'status':'failed' if failures else 'complete' if exhausted and not pending else 'running'}
     if failures:status['errors']=failures
     (r/'progress.json').write_text(json.dumps(status,indent=2));print(dump(status),flush=True);last=now
    if failures and not pending:raise RuntimeError('; '.join(failures))

def infer(scope,limit=None):
 try:return _infer(scope,limit)
 except BaseException as error:
                                                                       
                                                                  
  r=root(scope);dest=r/'responses.jsonl';tokens=requests=0
  if dest.exists():
   for line in dest.open():
    record=json.loads(line);tokens+=record['response']['usage']['input_tokens'];requests+=1
  conclusions=r/'conclusions.jsonl'
  completed=sum(1 for _ in conclusions.open()) if conclusions.exists() else 0
  status={'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
          'completed_pairs':completed,'successful_requests':requests,'input_tokens':tokens,
          'estimated_cost_usd':tokens*PRICE_PER_TOKEN,'error':type(error).__name__+': '+str(error)}
  (r/'progress.json').write_text(json.dumps(status,indent=2))
  raise

def build():
 r=root('evaluation');rows=[json.loads(l) for l in (r/'conclusions.jsonl').open()]
 by={v['pair_id']:v for v in rows};template=pl.read_csv('data/sample_submission.csv');assert len(by)==len(rows)==len(template)==112540
 assert set(by)==set(template['pair_id'])
 out=Path('artifacts/candidate_jev_direct');out.mkdir(exist_ok=True)
 with (out/'submission.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=template.columns);writer.writeheader()
  for pid in template['pair_id']:
   a=by[pid];e=a['evidence'];assert len(set(e))==len(e)
   writer.writerow({'pair_id':pid,'risk_score':a['risk_score'],'predicted_behavior':a['predicted_behavior'],**{f'evidence_hand_{i+1}':e[i] if i<len(e) else 'NO_EVIDENCE' for i in range(5)}})
 pl.DataFrame(rows).write_parquet(out/'jev_conclusions_with_confidence.parquet')
 print('Built',out/'submission.csv',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','infer','build']);ap.add_argument('--scope',choices=['pilot','evaluation'],default='pilot');ap.add_argument('--limit',type=int);args=ap.parse_args()
 if args.stage=='prepare':prepare(args.scope)
 elif args.stage=='preflight':preflight(args.scope)
 elif args.stage=='infer':infer(args.scope,args.limit)
 else:build()
