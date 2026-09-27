\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import polars as pl
from jev_client import MODEL, PRICE_PER_TOKEN, credential, evaluate

ROOT = Path('artifacts/evidence_session197_jev_pilot')
C = pl.col
QUESTION_TEXT = {
    'donate_a': 'Does A make a strategically poor contribution or surrender that appears to transfer value specifically to B?',
    'donate_b': 'Does B make a strategically poor contribution or surrender that appears to transfer value specifically to A?',
    'protect_a': 'Does A forgo a plausible profitable bet or raise against B, showing preferentially gentle play toward B?',
    'protect_b': 'Does B forgo a plausible profitable bet or raise against A, showing preferentially gentle play toward A?',
    'pressure_a': 'Does A use strategically unusual aggression against other players that helps B contest the pot or excludes opponents for B?',
    'pressure_b': 'Does B use strategically unusual aggression against other players that helps A contest the pot or excludes opponents for A?',
    'private_a': 'Does A take an action that is difficult to justify from A\'s own information but becomes sensible when B\'s private cards are known?',
    'private_b': 'Does B take an action that is difficult to justify from B\'s own information but becomes sensible when A\'s private cards are known?',
    'ordinary': 'Are the actions of both A and B reasonably explained by ordinary independent poker strategy without preferential treatment or knowledge of the other player\'s private cards?',
}
QUESTIONS = {k: {'type': 'noul', 'instructions': (
    'Evaluate only the observed six-player no-limit Texas Holdem hand in `hand`. '
    'A and B are the candidate pair, not established collaborators. '
    'Private cards are retrospective analyst information, not information legitimately available to the other player. '
    'Consider the board visible at each action and ordinary bluffing, pot odds, position and stack constraints. '
    'Winning or losing chips alone does not establish preferential play. '
    + q), 'criteria': {'true': 'The observed action sequence supports this specific statement.',
                      'false': 'The sequence does not support this statement; do not invent unobserved actions or intentions.'}}
    for k, q in QUESTION_TEXT.items()}
CONFIG = {
    'model': MODEL, 'questions': QUESTIONS, 'seed': 197,
    'pairs_per_family_per_fold': 4, 'folds': 4,
    'concurrency': 4, 'spending_cap_usd': 3.0,
    'sampling': 'sha256(jev197 + pair_id), stratified family/fold, no error selection',
    'state': 'One complete hand with A/B hole cards, all public actions, computed ratios and cached per-street equity; no labels or model scores',
    'evaluation': 'All shared hands; full pair MAP@5; four existing pool folds; fixed matched CatBoost controls and fixed R33 residual transport',
    'catboost': {'iterations': 250, 'depth': 3, 'learning_rate': .035,
                 'l2_leaf_reg': 20, 'thread_count': 2, 'verbose': False,
                 'allow_writing_files': False},
    'limitations': ['Known public families only; no measured hidden-family transfer',
                   'Small balanced pilot; public folds were reused by earlier research',
                   'Unlisted hands can be censored positives, not true ordinary play',
                   'No leaderboard claim from local improvement'],
}


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def available(p):
    p = Path(p)
    if not p.exists() or p.stat().st_flags & 0x40000000:
        raise RuntimeError(f'Required file absent/offloaded: {p}')
    return p


def canonical():
    d = pl.read_parquet(available('artifacts/policy/evidence_training.parquet'))
    ix = pl.read_parquet(available('artifacts/evidence_session4/hand_index.parquet'))
    d = d.join(ix.select('pair_id', 'hand_id', 'fold'), on=['pair_id', 'hand_id'], validate='1:1')
    equal = pl.read_parquet(available('artifacts/evidence_session64_equal_list_pressure/oof.parquet'))
    fallback = pl.read_parquet(available('artifacts/evidence_session11/conditional_family_deployed_oof.parquet'))
    risk = pl.read_csv(available('artifacts/evidence_session9/routed_evidence.csv')).filter(C('window') == 'full')
    d = d.join(equal.select('pair_id', 'hand_id', 'equal'), on=['pair_id', 'hand_id'], validate='1:1')
    d = d.join(fallback.select('pair_id', 'hand_id', 'conditional_family'), on=['pair_id', 'hand_id'], validate='1:1')
    d = d.join(risk.select('pair_id', 'risk_score'), on='pair_id', validate='m:1')
    return d.with_columns(pl.when(C('risk_score') >= .05).then(C('equal'))
                         .otherwise(C('conditional_family')).alias('r33')).sort('pair_id', 'time', 'hand_id')


def strength(e):
    if e is None: return 'unavailable'
    return 'weak' if e < .35 else 'medium' if e < .6 else 'strong' if e < .8 else 'very strong'


def state_for(hand, seats, actions, equities, pair):
    p1, p2 = pair
    names = {p1: 'A', p2: 'B'}
    for row in sorted(seats, key=lambda r:r['seat_no']):
        if row['player_id'] not in names: names[row['player_id']] = 'opponent_' + str(row['seat_no'])
    bb = hand['big_blind']
    board = (hand['board_cards'] or '').split()
    players = []
    for row in sorted(seats, key=lambda r:r['seat_no']):
        name = names[row['player_id']]
        item = {'player': name, 'seat': row['seat_no'], 'button': row['seat_no'] == hand['button_seat'],
                'starting_stack_bb': round(row['starting_stack']/bb, 3)}
        if name in ('A','B'): item['hole_cards'] = [row['hole_card_1'], row['hole_card_2']]
        players.append(item)
    trace = []
    for row in sorted(actions, key=lambda r:r['action_no']):
        street = row['street']; street_no = {'preflop':0,'flop':1,'turn':2,'river':3}[street]
        name = names[row['player_id']]
        item = {'player': name, 'street': street, 'action': row['action'],
                'amount_bb': round(row['amount']/bb, 3), 'amount_to_bb': round(row['amount_to']/bb, 3),
                'pot_before_bb': round(row['pot_before']/bb,3), 'to_call_bb': round(row['to_call']/bb,3),
                'stack_before_bb': round(row['stack_before']/bb,3), 'players_active':row['players_active']}
        if row['to_call'] > 0:
            item['call_pot_odds'] = round(row['to_call']/max(1,row['pot_before']+row['to_call']),3)
        if name in ('A','B'):
            e = equities.get((row['player_id'],street_no))
            item['estimated_equity_vs_random_opponent'] = None if e is None else round(e,3)
            item['equity_band'] = strength(e)
        trace.append(item)
    return {'hand': {'game': 'no-limit Texas Holdem', 'players': players,
                     'public_board_by_street': {'preflop':[], 'flop':board[:3], 'turn':board[:4], 'river':board[:5]},
                     'actions':trace,
                     'equity_note':'Equity estimates use only that player\'s cards and board at the street, not partner cards; they are not exact action EVs.'}}


def prepare():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'requests.jsonl').exists(), 'Existing pilot inputs must not be overwritten'
    d = canonical()
    pairs = d.select('pair_id','table_id','fold','behavior_family').unique()
    assert pairs.group_by('table_id').agg(C('fold').n_unique())['fold'].max() == 1
    chosen = []
    for _, group in pairs.group_by('fold','behavior_family'):
        ids = sorted(group['pair_id'].to_list(), key=lambda p:hashlib.sha256(('jev197'+p).encode()).hexdigest())[:4]
        chosen.extend(ids)
    assert len(chosen) == 48
    d = d.filter(C('pair_id').is_in(chosen))
    labels = pl.read_csv('data/development_labels.csv')
    endpoints = {r['pair_id']:(r['player_1'],r['player_2']) for r in labels.filter(C('pair_id').is_in(chosen)).to_dicts()}
    hands = d['hand_id'].unique().to_list()
    raw = {}
    for name in ['hands','seats','actions']:
        raw[name] = pl.scan_parquet(available(f'data/{name}.parquet')).filter(C('hand_id').is_in(hands)).collect(engine='streaming')
    hm = {r['hand_id']:r for r in raw['hands'].to_dicts()}
    sm = {k[0]:g.to_dicts() for k,g in raw['seats'].group_by('hand_id')}
    am = {k[0]:g.to_dicts() for k,g in raw['actions'].group_by('hand_id')}
    em = {}
    for table in sorted(d['table_id'].unique()):
        p = available(f'artifacts/policy/states/{table}.parquet')
        q = pl.read_parquet(p,columns=['hand_id','player_id','street_no','equity']).filter(C('hand_id').is_in(hands))
        for r in q.to_dicts(): em.setdefault(r['hand_id'],{})[(r['player_id'],int(r['street_no']))] = r['equity']
    keys = d.select('pair_id','hand_id','table_id','fold','behavior_family','evidence','time','r33')
    keys.write_parquet(ROOT/'metadata.parquet')
    pairs.filter(C('pair_id').is_in(chosen)).sort('fold','behavior_family','pair_id').write_csv(ROOT/'pairs_manifest.csv')
    total_bytes=0; max_bytes=0
    with (ROOT/'requests.jsonl').open('w') as f:
        for r in keys.to_dicts():
            h=r['hand_id']; state=state_for(hm[h],sm[h],am[h],em[h],endpoints[r['pair_id']])
            payload={'model':MODEL,'state':state,'questions':QUESTIONS}
            payload_text=json.dumps(payload,separators=(',',':'),ensure_ascii=True)
            assert all(token not in payload_text for token in [r['pair_id'],h,'evidence_rank','risk_score','behavior_family'])
            n=len(payload_text.encode());total_bytes+=n;max_bytes=max(max_bytes,n)
            f.write(json.dumps({'pair_id':r['pair_id'],'hand_id':h,'request_sha256':digest(payload),'state':state})+'\n')
    cfg=dict(CONFIG,rows=len(keys),pairs=len(chosen),pools=d['table_id'].n_unique(),request_bytes=total_bytes,
             conservative_cost_upper_usd=total_bytes*PRICE_PER_TOKEN,max_request_bytes=max_bytes,
             input_sha256=hashlib.sha256((ROOT/'requests.jsonl').read_bytes()).hexdigest())
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2))
    assert max_bytes<30000, 'Unexpectedly large state'
    print(json.dumps({k:v for k,v in cfg.items() if k not in ('questions','catboost')}),flush=True)


def infer():
    cfg=json.loads((ROOT/'config.json').read_text())
    assert cfg['model']==MODEL and cfg['questions']==QUESTIONS
    assert hashlib.sha256((ROOT/'requests.jsonl').read_bytes()).hexdigest()==cfg['input_sha256']
    assert cfg['conservative_cost_upper_usd'] < cfg['spending_cap_usd'], 'Reduce pilot before API use'
    records=[json.loads(s) for s in (ROOT/'requests.jsonl').read_text().splitlines()]
    dest=ROOT/'responses.jsonl';done={}
    if dest.exists():
        for line in dest.read_text().splitlines():
            r=json.loads(line);done[r['request_sha256']]=r
    pending=[r for r in records if r['request_sha256'] not in done]
    if not pending: print('All responses cached');return
    key=credential();start=time.monotonic();tokens=sum(r['response']['usage']['input_tokens'] for r in done.values())
                                                                            
                                                                                 
    reserve=sum((len(json.dumps({'model':MODEL,'state':r['state'],'questions':QUESTIONS},separators=(',',':')).encode())+2000)*3 for r in pending)
                                                                               
                                                                      
    reserve=reserve//3
    assert (tokens+reserve)*PRICE_PER_TOKEN<=cfg['spending_cap_usd'], 'Conservative budget including overhead exceeded'
    def one(r):
        payload={'model':MODEL,'state':r['state'],'questions':QUESTIONS}
        assert digest(payload)==r['request_sha256']
        result,seconds=evaluate(key,payload,attempts=1)
        return {'pair_id':r['pair_id'],'hand_id':r['hand_id'],'request_sha256':r['request_sha256'],
                'response':result,'seconds':seconds}
    completed=0
    with dest.open('a',buffering=1) as f, ThreadPoolExecutor(max_workers=cfg['concurrency']) as pool:
                                                                                 
        for offset in range(0,len(pending),cfg['concurrency']):
            futures=[pool.submit(one,r) for r in pending[offset:offset+cfg['concurrency']]]
            errors=[]
            for future in as_completed(futures):
                try:r=future.result()
                except Exception as e:errors.append(type(e).__name__+': '+str(e));continue
                f.write(json.dumps(r)+'\n');completed+=1;tokens+=r['response']['usage']['input_tokens']
            if errors:raise RuntimeError('; '.join(errors))
            assert tokens*PRICE_PER_TOKEN<cfg['spending_cap_usd']
            if completed%100==0 or offset+cfg['concurrency']>=len(pending):
                status={'completed_new':completed,'total_cached':len(done)+completed,'total':len(records),
                        'input_tokens':tokens,'estimated_cost_usd':round(tokens*PRICE_PER_TOKEN,6),
                        'seconds':round(time.monotonic()-start,1)}
                (ROOT/'api_progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)


def compare():
    from catboost import CatBoostClassifier, Pool
    from scipy.special import expit, logit
    d=canonical().join(pl.read_parquet(ROOT/'metadata.parquet').select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi')
    rows=[]
    for line in (ROOT/'responses.jsonl').read_text().splitlines():
        r=json.loads(line);row={k:r[k] for k in ('pair_id','hand_id')}
        row.update({'jev_'+k:v['noul'] for k,v in r['response']['answers'].items()});rows.append(row)
    jcols=['jev_'+k for k in QUESTIONS]
    j=pl.DataFrame(rows);assert len(j)==len(d) and j.select('pair_id','hand_id').n_unique()==len(d)
    d=d.join(j,on=['pair_id','hand_id'],validate='1:1')
    g=pl.read_parquet(available('artifacts/evidence_session195_grounded_selection/grounded.parquet'))
    gc=json.loads(Path('artifacts/evidence_session195_grounded_selection/columns.json').read_text())
    d=d.join(g,on=['pair_id','hand_id'],validate='1:1').sort('pair_id','time','hand_id')
    basic=['relative_time','pot','team_net','net_direction','both_showdown','both_fold','both_survive','seat_distance']
    x=d.select(gc+basic).to_numpy().astype(np.float32);x=np.nan_to_num(x,nan=0,posinf=1e6,neginf=-1e6)
    jx=d.select(jcols).to_numpy().astype(np.float32)
    y=d['evidence'].to_numpy().astype(int);fold=d['fold'].to_numpy()
    pred={'r33':d['r33'].to_numpy(),'jev_max':jx[:,:8].max(1)}
    for name in ['control','jev_augmented','r33_transport']:pred[name]=np.zeros(len(d))
    audits=[]
    for f in range(4):
        prior=d.select('pair_id','hand_id').join(pl.read_parquet(available(f'artifacts/evidence_session66_current_candidates/design_fold{f}.parquet')).select('pair_id','hand_id','prior'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['prior'].to_numpy()
        p0=logit(prior.clip(1e-6,1-1e-6));tr=fold!=f;va=fold==f
        train_pools=set(d.filter(pl.Series(tr))['table_id']);val_pools=set(d.filter(pl.Series(va))['table_id'])
        assert not train_pools&val_pools
        raw={}
        for name,xx in [('control',x),('jev_augmented',np.column_stack([x,jx]))]:
            m=CatBoostClassifier(**CONFIG['catboost'],random_seed=1970+f)
            m.fit(Pool(xx[tr],y[tr],baseline=p0[tr]))
            m.save_model(str(ROOT/f'{name}_fold{f}.cbm'))
            v=m.predict(xx[va],prediction_type='RawFormulaVal',thread_count=2);raw[name]=v
            reload=CatBoostClassifier();reload.load_model(str(ROOT/f'{name}_fold{f}.cbm'))
            np.testing.assert_array_equal(v,reload.predict(xx[va],prediction_type='RawFormulaVal',thread_count=2))
            pred[name][va]=expit(p0[va]+v)
                                                                          
                                                                        
        pred['r33_transport'][va]=expit(logit(pred['r33'][va].clip(1e-6,1-1e-6))+raw['jev_augmented']-raw['control'])
        audits.append({'fold':f,'train_pools':len(train_pools),'heldout_pools':len(val_pools),'overlap':0,'checkpoint_replay_exact':True})
    pairrows=[]
    d=d.with_row_index('row')
    for _,group in d.group_by('pair_id',maintain_order=True):
        ix=group['row'].to_numpy();truth=y[ix];den=min(5,int(truth.sum()));assert den>0
        row={k:group[k][0] for k in ['pair_id','table_id','fold','behavior_family']}
        for name,p in pred.items():
            order=np.lexsort((group['hand_id'].to_numpy(),-p[ix]))[:5];z=truth[order]
            row[name]=float((z*np.cumsum(z)/np.arange(1,len(z)+1)).sum()/den)
        pairrows.append(row)
    pairs=pl.DataFrame(pairrows);pairs.write_csv(ROOT/'pair_metrics.csv')
    d.select('pair_id','hand_id','fold','evidence',*jcols).with_columns([pl.Series(k,v) for k,v in pred.items()]).write_parquet(ROOT/'oof.parquet')
    pool=pairs.group_by('table_id').agg(C(list(pred)).sum(),pl.len().alias('n'))
    boot=np.random.default_rng(197).integers(0,len(pool),(5000,len(pool)));report={}
    for name in pred:
        delta=pool[name].to_numpy()-pool['r33'].to_numpy();bs=delta[boot].sum(1)/pool['n'].to_numpy()[boot].sum(1)
        report[name]={'MAP':pairs[name].mean(),'delta_vs_r33':pairs[name].mean()-pairs['r33'].mean(),
                      'pool_bootstrap_CI95':np.quantile(bs,[.025,.975]).tolist(),
                      'by_family':dict(pairs.group_by('behavior_family').agg(C(name).mean()).iter_rows()),
                      'by_fold':dict(pairs.group_by('fold').agg(C(name).mean()).iter_rows())}
    delta=pool['jev_augmented'].to_numpy()-pool['control'].to_numpy();bs=delta[boot].sum(1)/pool['n'].to_numpy()[boot].sum(1)
    report['matched_jev_gain']={'delta':pairs['jev_augmented'].mean()-pairs['control'].mean(),'pool_bootstrap_CI95':np.quantile(bs,[.025,.975]).tolist()}
    (ROOT/'audit.json').write_text(json.dumps(audits,indent=2))
    (ROOT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','infer','compare']);args=ap.parse_args()
    {'prepare':prepare,'infer':infer,'compare':compare}[args.stage]()
