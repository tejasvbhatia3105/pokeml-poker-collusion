\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import argparse,gzip,hashlib,json
from pathlib import Path
import polars as pl
import session197_jev_pilot as p
import session198_jev_full as full
from jev_client import PRICE_PER_TOKEN

ROOT=Path('artifacts/evidence_session199_jev_evaluation');C=pl.col
BASE=Path('artifacts/candidate_r54_maxep70/submission.csv')


def prepare():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'requests.jsonl.gz').exists()
    base=pl.read_csv(p.available(BASE));pairs=pl.read_csv(p.available('data/evaluation_pairs.csv'))
    pairs=pairs.join(base.select('pair_id','risk_score','predicted_behavior'),on='pair_id',validate='1:1').filter(C('risk_score')>=.01)
    hands=pl.scan_parquet(p.available('data/hands.parquet')).filter(C('phase')=='evaluation').collect()
    seats=pl.scan_parquet(p.available('data/seats.parquet')).join(hands.lazy().select('hand_id'),on='hand_id',how='semi').collect(engine='streaming')
    keys=pairs.join(seats.select('hand_id',C('player_id').alias('player_1')),on='player_1')
    keys=keys.join(seats.select('hand_id',C('player_id').alias('player_2')),on=['hand_id','player_2'])
    keys=keys.join(hands.select('hand_id','table_id','started_at'),on='hand_id').sort('pair_id','started_at','hand_id')
    assert keys.select('pair_id','hand_id').n_unique()==len(keys)
    keys.write_parquet(ROOT/'metadata.parquet')
    endpoints={r['pair_id']:(r['player_1'],r['player_2']) for r in pairs.to_dicts()}
    hand_ids=keys['hand_id'].unique().to_list()
    hands=hands.filter(C('hand_id').is_in(hand_ids));seats=seats.filter(C('hand_id').is_in(hand_ids))
    actions=pl.scan_parquet(p.available('data/actions.parquet')).filter(C('hand_id').is_in(hand_ids)).collect(engine='streaming')
    hm={r['hand_id']:r for r in hands.to_dicts()}
    count=0;batches=0;totalbytes=0;maxbytes=0;buffer=[]
    with gzip.open(ROOT/'requests.jsonl.gz','wt',compresslevel=3) as f:
        for table,group in keys.group_by('table_id',maintain_order=True):
            table=table[0];hs=group['hand_id'].unique().to_list()
            sm={k[0]:g.to_dicts() for k,g in seats.filter(C('hand_id').is_in(hs)).group_by('hand_id')}
            am={k[0]:g.to_dicts() for k,g in actions.filter(C('hand_id').is_in(hs)).group_by('hand_id')}
            eq=pl.read_parquet(p.available(f'artifacts/policy/states/{table}.parquet'),columns=['hand_id','player_id','street_no','equity']).filter(C('hand_id').is_in(hs))
            em={}
            for row in eq.to_dicts():em.setdefault(row['hand_id'],{})[(row['player_id'],int(row['street_no']))]=row['equity']
            for row in group.to_dicts():
                h=row['hand_id'];state=p.state_for(hm[h],sm[h],am[h],em[h],endpoints[row['pair_id']])
                buffer.append({'pair_id':row['pair_id'],'hand_id':h,'state':state});count+=1
                if len(buffer)==4:
                    payload=full.payload_for(buffer);n=len(json.dumps(payload,separators=(',',':')).encode());totalbytes+=n;maxbytes=max(n,maxbytes)
                    f.write(json.dumps({'request_sha256':p.digest(payload),'rows':buffer})+'\n');batches+=1;buffer=[]
            if count%10000<500:print('prepared evaluation hands',count,flush=True)
        if buffer:
            payload=full.payload_for(buffer);n=len(json.dumps(payload,separators=(',',':')).encode());totalbytes+=n;maxbytes=max(n,maxbytes)
            f.write(json.dumps({'request_sha256':p.digest(payload),'rows':buffer})+'\n');batches+=1
    assert count==len(keys)
    cfg=dict(full.CONFIG,scope='All shared evaluation hands for frozen r54 risk >=.01',sampling='No outcome labels; fixed risk gate .01',
             rows=count,batches=batches,pairs=keys['pair_id'].n_unique(),pools=keys['table_id'].n_unique(),request_bytes=totalbytes,
             max_request_bytes=maxbytes,spending_cap_usd=80,
             conservative_cost_upper_usd=(totalbytes+2000*batches)*PRICE_PER_TOKEN,
             input_sha256=hashlib.sha256((ROOT/'requests.jsonl.gz').read_bytes()).hexdigest(),
             base_submission=str(BASE),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest())
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    print(json.dumps({k:v for k,v in cfg.items() if k not in ('questions','catboost')}),flush=True)


def export():
    full.ROOT=ROOT;full.flatten();rows=[]
    for line in (ROOT/'responses.jsonl').open():
        r=json.loads(line);row={k:r[k] for k in ('pair_id','hand_id')}
        row.update({'jev_'+k:v['noul'] for k,v in r['response']['answers'].items()});rows.append(row)
    q=pl.DataFrame(rows);m=pl.read_parquet(ROOT/'metadata.parquet')
    assert len(q)==len(m)
    m.join(q,on=['pair_id','hand_id'],validate='1:1').write_parquet(ROOT/'features.parquet')
    print('Exported reusable evaluation features',len(q),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','infer','export']);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='infer':full.ROOT=ROOT;full.infer()
    else:export()
