\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import gzip,hashlib,json
from itertools import zip_longest
from pathlib import Path
import polars as pl
import session197_jev_pilot as pilot
import session198_jev_full as batch
from jev_client import MODEL,PRICE_PER_TOKEN

ROOT=Path('artifacts/evidence_session206_jev_final_audit')


def study(directory,single=False):
    root=Path('artifacts')/directory;cfg=json.loads((root/'config.json').read_text())
    src=root/('requests.jsonl' if single else 'requests.jsonl.gz')
    assert hashlib.sha256(src.read_bytes()).hexdigest()==cfg['input_sha256']
    responses=root/('responses.jsonl' if single else 'batch_responses.jsonl')
    actual={};tokens=0;answered=set();response_digests={}
    for line in responses.open():
        r=json.loads(line);sha=r['request_sha256'];assert sha not in actual
        keys=[{k:r[k] for k in ('pair_id','hand_id')}] if single else r['keys']
        response=r['response'];assert response['model']==MODEL
        names=set(pilot.QUESTIONS) if single else {f'h{i}_{k}' for i in range(len(keys)) for k in pilot.QUESTIONS}
        assert set(response['answers'])==names
        assert all(v['type']=='noul' and 0<=v['noul']<=1 for v in response['answers'].values())
        assert isinstance(response['usage']['input_tokens'],int) and response['usage']['input_tokens']>0
        tokens+=response['usage']['input_tokens'];actual[sha]=keys
        response_digests[sha]=pilot.digest(r)
        for key in keys:
            pair=key['pair_id'],key['hand_id'];assert pair not in answered;answered.add(pair)
    expected=set();rows=0
    with (src.open() if single else gzip.open(src,'rt')) as source:
        for line in source:
            r=json.loads(line)
            payload={'model':MODEL,'state':r['state'],'questions':pilot.QUESTIONS} if single else batch.payload_for(r['rows'])
            sha=pilot.digest(payload);assert sha==r['request_sha256'] and sha not in expected;expected.add(sha)
            keys=[{k:r[k] for k in ('pair_id','hand_id')}] if single else [{k:x[k] for k in ('pair_id','hand_id')} for x in r['rows']]
            assert actual.get(sha)==keys
            rows+=len(keys)
    assert set(actual)==expected and rows==cfg['rows']
    meta=pl.read_parquet(root/'metadata.parquet');assert meta.select('pair_id','hand_id').n_unique()==len(meta)
    assert set(meta.select('pair_id','hand_id').iter_rows())<=answered
    borrowed_tokens=0;borrowed_requests=0
    if 'borrowed_cache_source' in cfg:
        borrowed=Path(cfg['borrowed_cache_source'])
        assert hashlib.sha256(borrowed.read_bytes()).hexdigest()==cfg['borrowed_cache_sha256']
        borrowed_seen=set()
        for line in borrowed.open():
            r=json.loads(line);sha=r['request_sha256'];assert sha not in borrowed_seen
            borrowed_seen.add(sha)
            assert response_digests.get(sha)==pilot.digest(r)
            borrowed_tokens+=r['response']['usage']['input_tokens'];borrowed_requests+=1
        assert borrowed_tokens==cfg['borrowed_input_tokens'] and borrowed_requests==cfg['borrowed_requests']
    return {'api_observations':rows,'scored_or_exported_observations':len(meta),'requests':len(actual),
            'input_tokens':tokens,'estimated_successful_cost_usd':tokens*PRICE_PER_TOKEN,
            'borrowed_requests':borrowed_requests,'borrowed_input_tokens':borrowed_tokens,
            'new_successful_requests':len(actual)-borrowed_requests,
            'estimated_incremental_cost_usd':(tokens-borrowed_tokens)*PRICE_PER_TOKEN,
            'request_sha256':cfg['input_sha256'],'response_sha256':hashlib.sha256(responses.read_bytes()).hexdigest(),
            'exact_request_response_keys':True,'all_answers_valid':True}


def main():
    ROOT.mkdir(exist_ok=True)
    results={}
    for n,single in [('197_jev_pilot',True),('198_jev_full',False),('199_jev_evaluation',False),('204_jev_shared_prefix',False),('209_jev_full_prefix',False)]:
        results[n]=study('evidence_session'+n,single);print('verified',n,flush=True)
    from session204_jev_shared_prefix import focused
    original=Path('artifacts/evidence_session198_jev_full/requests.jsonl.gz')
    prefix=Path('artifacts/evidence_session209_jev_full_prefix/requests.jsonl.gz')
    with gzip.open(original,'rt') as a,gzip.open(prefix,'rt') as b:
        for old,new in zip_longest(a,b):
            assert old is not None and new is not None
            old=json.loads(old);new=json.loads(new)
            expected=[dict(r,state=focused(r['state'])) for r in old['rows']]
            assert new['rows']==expected
    print('verified exact prefix transform and unchanged batch membership',flush=True)
    evaluation=pl.read_parquet('artifacts/evidence_session199_jev_evaluation/features.parquet')
    meta=pl.read_parquet('artifacts/evidence_session199_jev_evaluation/metadata.parquet')
    assert evaluation.select('pair_id','hand_id').sort('pair_id','hand_id').equals(meta.select('pair_id','hand_id').sort('pair_id','hand_id'))
    cols=['jev_'+k for k in pilot.QUESTIONS]
    assert evaluation.select(pl.col(cols).is_null().sum()).row(0)==(0,)*len(cols)
    assert evaluation.select(pl.col(cols).is_between(0,1).all()).row(0)==(True,)*len(cols)
    best={
      'artifacts/candidate_r54_maxep70/submission.csv':'0b17e6b6f158893900ead753d222204105827075a72376b3d44f0bc76388a124',
      'artifacts/candidate_r33/submission.csv':'da627814c7474862c032b6cd28189dc55741eb50f92f47f2a3355c1c78051904'}
    for file,sha in best.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha
    out={'studies':results,'estimated_successful_cost_usd':sum(v['estimated_incremental_cost_usd'] for v in results.values()),
         'new_successful_requests':sum(v['new_successful_requests'] for v in results.values()),
         'billing_caveat':'Returned successful-request token usage times published price; excludes tiny synthetic smoke test and any unreported failed-request billing.',
         'evaluation_features_complete':True,'preserved_submissions':best,
         'prefix_transform_exact':True,
         'unique_primary_scope_observations':45129+127594,
         'scope_caveat':'All public positive-pair shared hands plus all shared evaluation hands at frozen r54 risk>=.01. Not all two million raw hands or all evaluation pairs.'}
    (ROOT/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
