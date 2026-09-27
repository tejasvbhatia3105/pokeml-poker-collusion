\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,gzip,hashlib,json,threading,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import polars as pl
import session197_jev_pilot as pilot
from jev_client import MODEL,PRICE_PER_TOKEN,credential,evaluate

ROOT=Path('artifacts/evidence_session198_jev_full')
C=pl.col
CONFIG=dict(pilot.CONFIG,scope='All45129public shared hands,372labelled pairs',
            batch_hands=4,concurrency=8,max_requests_per_second=15,spending_cap_usd=30,
            sampling='All canonical public evidence pairs and all their shared hands',
            pairs_per_family_per_fold=None,
            limitations=['Known public families only; hidden-family transfer is not measurable here',
                         'Public pool folds reused heavily by prior research',
                         'Unlisted hands can be censored positives, not true ordinary play',
                         'Four-hand context can shift judgments versus the single-hand pilot',
                         'No leaderboard claim from local improvement'],
            method='Same nine atomic questions; explicit per-hand JSON paths; matched fixed models')


def payload_for(rows):
    questions={}
    for i in range(len(rows)):
        for k,q in pilot.QUESTIONS.items():
            questions[f'h{i}_{k}']=dict(q,instructions=q['instructions'].replace('`hand`',f'`hands[{i}]`'))
    return {'model':MODEL,'state':{'hands':[r['state']['hand'] for r in rows]},'questions':questions}


def prepare():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'requests.jsonl.gz').exists(),'Do not overwrite frozen requests'
    d=pilot.canonical();assert len(d)==45129
    keys=d.select('pair_id','hand_id','table_id','fold','behavior_family','evidence','time','r33')
    keys.write_parquet(ROOT/'metadata.parquet')
    endpoints={r['pair_id']:(r['player_1'],r['player_2']) for r in pl.read_csv('data/development_labels.csv').filter(C('label')==1).to_dicts()}
    hand_ids=d['hand_id'].unique().to_list();raw={}
    for name in ['hands','seats','actions']:
        raw[name]=pl.scan_parquet(pilot.available(f'data/{name}.parquet')).filter(C('hand_id').is_in(hand_ids)).collect(engine='streaming')
    hm={r['hand_id']:r for r in raw['hands'].to_dicts()}
                                                                        
    count=0;batches=0;totalbytes=0;maxbytes=0;buffer=[]
    with gzip.open(ROOT/'requests.jsonl.gz','wt',compresslevel=3) as f:
        for table,group in keys.group_by('table_id',maintain_order=True):
            table=table[0];hs=group['hand_id'].unique().to_list()
            sm={k[0]:g.to_dicts() for k,g in raw['seats'].filter(C('hand_id').is_in(hs)).group_by('hand_id')}
            am={k[0]:g.to_dicts() for k,g in raw['actions'].filter(C('hand_id').is_in(hs)).group_by('hand_id')}
            eq=pl.read_parquet(pilot.available(f'artifacts/policy/states/{table}.parquet'),columns=['hand_id','player_id','street_no','equity']).filter(C('hand_id').is_in(hs))
            em={}
            for row in eq.to_dicts():em.setdefault(row['hand_id'],{})[(row['player_id'],int(row['street_no']))]=row['equity']
            for row in group.to_dicts():
                h=row['hand_id'];state=pilot.state_for(hm[h],sm[h],am[h],em[h],endpoints[row['pair_id']])
                buffer.append({'pair_id':row['pair_id'],'hand_id':h,'state':state});count+=1
                if len(buffer)==CONFIG['batch_hands']:
                    payload=payload_for(buffer);n=len(json.dumps(payload,separators=(',',':')).encode());totalbytes+=n;maxbytes=max(n,maxbytes)
                    f.write(json.dumps({'request_sha256':pilot.digest(payload),'rows':buffer})+'\n');batches+=1;buffer=[]
            if count%1000<200:print('prepared',count,'hands',flush=True)
        if buffer:
            payload=payload_for(buffer);n=len(json.dumps(payload,separators=(',',':')).encode());totalbytes+=n;maxbytes=max(n,maxbytes)
            f.write(json.dumps({'request_sha256':pilot.digest(payload),'rows':buffer})+'\n');batches+=1
    assert count==45129
    cfg=dict(CONFIG,rows=count,batches=batches,pairs=keys['pair_id'].n_unique(),pools=keys['table_id'].n_unique(),
             request_bytes=totalbytes,max_request_bytes=maxbytes,conservative_cost_upper_usd=(totalbytes+2000*batches)*PRICE_PER_TOKEN,
             input_sha256=hashlib.sha256((ROOT/'requests.jsonl.gz').read_bytes()).hexdigest())
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    assert cfg['conservative_cost_upper_usd']<cfg['spending_cap_usd']
    print(json.dumps({k:v for k,v in cfg.items() if k not in ('questions','catboost')}),flush=True)


def infer():
    cfg=json.loads((ROOT/'config.json').read_text());assert cfg['model']==MODEL and cfg['questions']==pilot.QUESTIONS
    assert hashlib.sha256((ROOT/'requests.jsonl.gz').read_bytes()).hexdigest()==cfg['input_sha256']
    dest=ROOT/'batch_responses.jsonl';done=set();tokens=0;nhands=0
    if dest.exists():
        for line in dest.open():
            r=json.loads(line);assert r['request_sha256'] not in done;done.add(r['request_sha256'])
            tokens+=r['response']['usage']['input_tokens'];nhands+=len(r['keys'])
    remaining=cfg['batches']-len(done)
    if not remaining:print('All batches cached');return
    key=credential();start=time.monotonic();lock=threading.Lock();next_start=[0.0]
                                                                         
                                                                               
    def one(r):
        payload=payload_for(r['rows']);assert pilot.digest(payload)==r['request_sha256']
        with lock:
            now=time.monotonic();wait=max(0,next_start[0]-now);next_start[0]=max(now,next_start[0])+1/cfg['max_requests_per_second']
        if wait:time.sleep(wait)
        result,seconds=evaluate(key,payload,attempts=3)
        return {'request_sha256':r['request_sha256'],'keys':[{k:x[k] for k in ('pair_id','hand_id')} for x in r['rows']],
                'response':result,'seconds':seconds}
    last_report=0
    with dest.open('a',buffering=1) as out,ThreadPoolExecutor(max_workers=cfg['concurrency']) as pool,gzip.open(ROOT/'requests.jsonl.gz','rt') as source:
        pending={};failures=[];exhausted=False
        while pending or not exhausted:
                                                                             
                                                                               
            while not failures and not exhausted and len(pending)<cfg['concurrency']:
                line=next(source,None)
                if line is None:exhausted=True;break
                r=json.loads(line)
                if r['request_sha256'] in done:continue
                reserve=3*(len(json.dumps(payload_for(r['rows']),separators=(',',':')).encode())+2000)
                assert (tokens+reserve+sum(pending.values()))*PRICE_PER_TOKEN<cfg['spending_cap_usd'],'Spending cap reached'
                pending[pool.submit(one,r)]=reserve
            if not pending:break
            ready,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for future in ready:
                pending.pop(future)
                try:r=future.result()
                except Exception as e:failures.append(type(e).__name__+': '+str(e));continue
                out.write(json.dumps(r)+'\n');done.add(r['request_sha256'])
                tokens+=r['response']['usage']['input_tokens'];nhands+=len(r['keys'])
            now=time.monotonic()
            if now-last_report>20 or len(done)==cfg['batches'] or (failures and not pending):
                progress={'hands_complete':nhands,'hands_total':cfg['rows'],'batches_complete':len(done),
                          'input_tokens':tokens,'estimated_cost_usd':round(tokens*PRICE_PER_TOKEN,6),'seconds_this_run':round(now-start,1),
                          'status':'failed' if failures and not pending else 'complete' if len(done)==cfg['batches'] else 'running'}
                if failures:progress['errors']=failures
                (ROOT/'api_progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True);last_report=now
            if failures and not pending:raise RuntimeError('; '.join(failures))


def flatten():
    cfg=json.loads((ROOT/'config.json').read_text());n=0;seen=set()
    with (ROOT/'responses.jsonl').open('w') as out:
        for line in (ROOT/'batch_responses.jsonl').open():
            batch=json.loads(line)
            for i,key in enumerate(batch['keys']):
                kk=(key['pair_id'],key['hand_id']);assert kk not in seen;seen.add(kk)
                answers={k:batch['response']['answers'][f'h{i}_{k}'] for k in pilot.QUESTIONS}
                                                                                
                out.write(json.dumps(dict(key,batch_sha256=batch['request_sha256'],response={'model':MODEL,'answers':answers}))+'\n');n+=1
    assert n==cfg['rows'];print('flattened',n,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','infer','compare']);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='infer':infer()
    else:
        flatten();pilot.ROOT=ROOT;pilot.compare()
