\
\
\
\
import gzip,hashlib,json,tempfile,time
from pathlib import Path
import session198_jev_full as runner


def main():
    original=(runner.ROOT,runner.evaluate,runner.credential)
    with tempfile.TemporaryDirectory(prefix='jev_runner_test_') as directory:
        runner.ROOT=Path(directory);records=[]
        for i in range(16):
            rows=[{'pair_id':str(i),'hand_id':str(i),'state':{'hand':{'index':i}}}]
            records.append({'request_sha256':runner.pilot.digest(runner.payload_for(rows)),'rows':rows})
        request=runner.ROOT/'requests.jsonl.gz'
        with gzip.open(request,'wt') as f:
            for r in records:f.write(json.dumps(r)+'\n')
        cfg=dict(runner.CONFIG,rows=16,batches=16,max_requests_per_second=1000,
                 input_sha256=hashlib.sha256(request.read_bytes()).hexdigest())
        (runner.ROOT/'config.json').write_text(json.dumps(cfg))
        calls=[];failed=[False]
        def fake(key,payload,attempts):
            i=payload['state']['hands'][0]['index'];calls.append(i)
            if i==2 and not failed[0]:
                failed[0]=True;raise RuntimeError('injected timeout')
            time.sleep(.02)
            return {'model':runner.MODEL,'answers':{k:{'type':'noul','noul':.5} for k in payload['questions']},'usage':{'input_tokens':10}},.02
        runner.evaluate=fake;runner.credential=lambda:'synthetic'
        try:
            try:runner.infer()
            except RuntimeError as e:assert str(e)=='RuntimeError: injected timeout'
            else:raise AssertionError('Expected injected error')
            saved=runner.ROOT/'batch_responses.jsonl'
            first=[json.loads(l) for l in saved.open()];assert len(first)==7
            progress=json.loads((runner.ROOT/'api_progress.json').read_text())
            assert progress['status']=='failed' and progress['hands_complete']==7
            runner.infer();all_rows=[json.loads(l) for l in saved.open()]
            assert len(all_rows)==16 and len({r['request_sha256'] for r in all_rows})==16
            assert calls.count(2)==2 and all(calls.count(i)==1 for i in range(16) if i!=2)
            before=len(calls);runner.infer();assert len(calls)==before
            assert json.loads((runner.ROOT/'api_progress.json').read_text())['status']=='complete'
        finally:runner.ROOT,runner.evaluate,runner.credential=original
    print('PASS: failed-run status, in-flight drain, exact resume, cached no-op')


if __name__=='__main__':main()
