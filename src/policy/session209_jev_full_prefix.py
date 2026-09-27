\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import argparse,copy,gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np,polars as pl
import session197_jev_pilot as pilot
import session198_jev_full as full
from session204_jev_shared_prefix import focused

ROOT=Path('artifacts/evidence_session209_jev_full_prefix')
SOURCE=Path('artifacts/evidence_session198_jev_full')
PREFIX=Path('artifacts/evidence_session204_jev_shared_prefix');C=pl.col


def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'requests.jsonl.gz').exists()
    count=0;batches=0
    with gzip.open(ROOT/'requests.jsonl.gz','wt',compresslevel=3) as out:
        for line in gzip.open(SOURCE/'requests.jsonl.gz','rt'):
            r=json.loads(line)
            for row in r['rows']:row['state']=focused(row['state'])
            r['request_sha256']=pilot.digest(full.payload_for(r['rows']))
            out.write(json.dumps(r)+'\n');count+=len(r['rows']);batches+=1
    shutil.copyfile(SOURCE/'metadata.parquet',ROOT/'metadata.parquet')
    shutil.copyfile(PREFIX/'batch_responses.jsonl',ROOT/'batch_responses.jsonl')
    borrowed=[json.loads(l) for l in (PREFIX/'batch_responses.jsonl').open()]
    meta=pl.read_parquet(SOURCE/'metadata.parquet')
    pilot_pools=set(pl.read_parquet(PREFIX/'metadata.parquet')['table_id'])
    pilot_pools.update(meta.filter(C('pair_id')=='P012927D10E56')['table_id'])
    cfg=dict(full.CONFIG,scope=__doc__,rows=count,batches=batches,max_requests_per_second=18,spending_cap_usd=6,
        input_sha256=hashlib.sha256((ROOT/'requests.jsonl.gz').read_bytes()).hexdigest(),
        borrowed_cache_source=str(PREFIX/'batch_responses.jsonl'),
        borrowed_cache_sha256=hashlib.sha256((PREFIX/'batch_responses.jsonl').read_bytes()).hexdigest(),
        borrowed_requests=len(borrowed),borrowed_input_tokens=sum(r['response']['usage']['input_tokens'] for r in borrowed),
        excluded_exploratory_pools=sorted(pilot_pools))
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    print(json.dumps({k:cfg[k] for k in ['rows','batches','borrowed_requests','borrowed_input_tokens','spending_cap_usd']}))


def compare():
    full.ROOT=ROOT;full.flatten();pilot.ROOT=ROOT;pilot.compare()
    excluded=json.loads((ROOT/'config.json').read_text())['excluded_exploratory_pools']
    a=pl.read_csv(ROOT/'pair_metrics.csv');b=pl.read_csv(SOURCE/'pair_metrics.csv')
    q=a.join(b.select('pair_id',C('r33_transport').alias('full_hand_transport')),on='pair_id',validate='1:1')
    results={}
    for name,g in [('all',q),('outside_pilot_and_inspected_pools',q.filter(~C('table_id').is_in(excluded)))]:
        pool=g.group_by('table_id').agg(C('r33','r33_transport','full_hand_transport').sum(),pl.len().alias('n'))
        ix=np.random.default_rng(209).integers(0,len(pool),(5000,len(pool)))
        r={'pairs':len(g),'pools':len(pool),'prefix_MAP':g['r33_transport'].mean(),'r33_MAP':g['r33'].mean(),'full_hand_transport_MAP':g['full_hand_transport'].mean()}
        for base in ['r33','full_hand_transport']:
            delta=pool['r33_transport'].to_numpy()-pool[base].to_numpy();bs=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1)
            r['vs_'+base]={'gain':g['r33_transport'].mean()-g[base].mean(),'pool_CI95':np.quantile(bs,[.025,.975]).tolist()}
        results[name]=r
    (ROOT/'scope_comparison.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','infer','compare']);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='infer':full.ROOT=ROOT;full.infer()
    else:compare()
