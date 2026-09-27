\
\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import argparse,copy,gzip,hashlib,json
from pathlib import Path
import polars as pl
import session197_jev_pilot as p
import session198_jev_full as full
ROOT=Path('artifacts/evidence_session204_jev_shared_prefix')
CONTROL=Path('artifacts/evidence_session205_jev_batched_pilot_control')
SOURCE=Path('artifacts/evidence_session198_jev_full')


def focused(state):
    state=copy.deepcopy(state);h=state['hand'];acts=[]
    for a in h['actions']:
        acts.append(a)
        if a['player'] in ('A','B') and a['action']=='fold':break
    h['actions']=acts
    last={'preflop':0,'flop':1,'turn':2,'river':3}[acts[-1]['street']] if acts else 0
    for i,street in enumerate(['preflop','flop','turn','river']):
        if i>last:h['public_board_by_street'][street]=[]
    h['scope_note']='Only the action prefix through the first fold by A or B is provided, or the full trace if neither folds. Do not infer omitted later actions.'
    return state


def prepare():
    ROOT.mkdir(exist_ok=True);CONTROL.mkdir(exist_ok=True)
    assert not (ROOT/'requests.jsonl.gz').exists()
    meta=pl.read_parquet('artifacts/evidence_session197_jev_pilot/metadata.parquet')
    wanted=set(meta.select('pair_id','hand_id').iter_rows());n=0;batches=0;totalbytes=0;changed=0;selected=set()
    with gzip.open(ROOT/'requests.jsonl.gz','wt',compresslevel=3) as out:
        for line in gzip.open(SOURCE/'requests.jsonl.gz','rt'):
            old=json.loads(line)
            if not any((r['pair_id'],r['hand_id']) in wanted for r in old['rows']):continue
            selected.add(old['request_sha256']);rows=copy.deepcopy(old['rows'])
            for r in rows:
                s=focused(r['state']);changed+=len(s['hand']['actions'])!=len(r['state']['hand']['actions']);r['state']=s
            payload=full.payload_for(rows);totalbytes+=len(json.dumps(payload,separators=(',',':')).encode())
            out.write(json.dumps({'request_sha256':p.digest(payload),'rows':rows})+'\n');n+=len(rows);batches+=1
    cfg=dict(full.CONFIG,scope=__doc__,rows=n,scored_rows=len(meta),batches=batches,concurrency=2,max_requests_per_second=1,
             spending_cap_usd=3,input_sha256=hashlib.sha256((ROOT/'requests.jsonl.gz').read_bytes()).hexdigest(),
             request_bytes=totalbytes,shortened_hands=changed,control=str(CONTROL))
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2));meta.write_parquet(ROOT/'metadata.parquet');meta.write_parquet(CONTROL/'metadata.parquet')
                                                                            
    with (CONTROL/'batch_responses.jsonl').open('w') as out:
        for line in (SOURCE/'batch_responses.jsonl').open():
            if json.loads(line)['request_sha256'] in selected:out.write(line)
    control_cfg=dict(cfg,scope='Exact198 batch responses restricted to the48 pilot pairs',
                     source_directory=str(SOURCE),
                     cached_responses_sha256=hashlib.sha256((CONTROL/'batch_responses.jsonl').read_bytes()).hexdigest())
    control_cfg.pop('input_sha256')                                                                  
    (CONTROL/'config.json').write_text(json.dumps(control_cfg,indent=2))
    print(json.dumps({k:cfg[k] for k in ['rows','scored_rows','batches','request_bytes','shortened_hands']}))


def flatten_scored(root):
    wanted=set(pl.read_parquet(root/'metadata.parquet').select('pair_id','hand_id').iter_rows());seen=set()
    with (root/'responses.jsonl').open('w') as out:
        for line in (root/'batch_responses.jsonl').open():
            b=json.loads(line)
            for i,key in enumerate(b['keys']):
                k=key['pair_id'],key['hand_id']
                if k not in wanted:continue
                assert k not in seen;seen.add(k)
                out.write(json.dumps(dict(key,response={'model':p.MODEL,'answers':{q:b['response']['answers'][f'h{i}_{q}'] for q in p.QUESTIONS}}))+'\n')
    assert seen==wanted


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','infer','compare']);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='infer':full.ROOT=ROOT;full.infer()
    else:
        for root in [CONTROL,ROOT]:
            flatten_scored(root);p.ROOT=root;p.compare()
