\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import argparse
import gzip,json
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr
from session197_jev_pilot import QUESTIONS

PILOT = Path('artifacts/evidence_session197_jev_pilot')
FULL = Path('artifacts/evidence_session198_jev_full')
OUT = Path('artifacts/evidence_session200_jev_audit')
KEYS = ['pair_id', 'hand_id']
COLS = ['jev_' + k for k in QUESTIONS]


def responses(root):
    rows = []
    for line in (root / 'responses.jsonl').open():
        r = json.loads(line)
        rows.append({**{k: r[k] for k in KEYS},
                     **{'jev_' + k: v['noul'] for k, v in r['response']['answers'].items()}})
    q = pl.DataFrame(rows)
    assert q.select(KEYS).n_unique() == len(q)
    return q


def discrimination(root):
    d = pl.read_parquet(root / 'metadata.parquet').join(responses(root), on=KEYS, validate='1:1')
    result = {}
    for family, g in [('all', d), *[(k[0], v) for k, v in d.group_by('behavior_family')]]:
        y = g['evidence'].to_numpy().astype(bool)
        n1 = int(y.sum()); n0 = len(y) - n1
        out = {}
        for col in COLS:
            x = g[col].to_numpy()
            auc = (rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            out[col] = {'auc_as_positive': float(auc), 'listed_mean': float(x[y].mean()),
                        'unlisted_mean': float(x[~y].mean()),
                        'distinct_values': int(np.unique(x).size)}
        result[family] = {'rows': len(g), 'listed': n1, 'judgments': out}
    return result


def batch_shift():
    a = responses(PILOT); b = responses(FULL)
    d = a.join(b, on=KEYS, suffix='_batch', validate='1:1')
    assert len(d) == len(a)
    return {'overlap_hands': len(d), 'questions': {
        c: {'mean_absolute_change': float(np.abs(d[c].to_numpy() - d[c + '_batch'].to_numpy()).mean()),
            'spearman': float(spearmanr(d[c].to_numpy(), d[c + '_batch'].to_numpy()).statistic),
            'fraction_change_over_0_2': float((np.abs(d[c].to_numpy() - d[c + '_batch'].to_numpy()) > .2).mean())}
        for c in COLS}}


def input_scope():
    rows=[]
    for line in gzip.open(FULL/'requests.jsonl.gz','rt'):
        for r in json.loads(line)['rows']:
            a=r['state']['hand']['actions']
            cut=next((i+1 for i,v in enumerate(a) if v['player'] in ('A','B') and v['action']=='fold'),len(a))
            rows.append({'pair_id':r['pair_id'],'hand_id':r['hand_id'],'actions_total':len(a),
                         'actions_prefix':cut,'actions_after_partner_fold':len(a)-cut})
    d=pl.DataFrame(rows).join(pl.read_parquet(FULL/'oof.parquet').select(*KEYS,'evidence','jev_max','r33'),on=KEYS,validate='1:1')
    d=d.join(pl.read_parquet(FULL/'metadata.parquet').select(*KEYS,'behavior_family'),on=KEYS,validate='1:1')
    d.write_parquet(OUT/'input_scope.parquet');result={}
    fields=['actions_total','actions_prefix','actions_after_partner_fold','jev_max']
    for family,g in d.group_by('behavior_family'):
        result[family[0]]={
            'rows':len(g),
            'jev_action_length_spearman':float(spearmanr(g['jev_max'],g['actions_total']).statistic),
            'jev_after_fold_length_spearman':float(spearmanr(g['jev_max'],g['actions_after_partner_fold']).statistic),
            'listed':g.filter(pl.col('evidence')==1).select(pl.col(fields).mean()).to_dicts()[0],
            'unlisted':g.filter(pl.col('evidence')==0).select(pl.col(fields).mean()).to_dicts()[0]}
    (OUT/'input_scope_report.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--full', action='store_true'); args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    result = {'pilot': discrimination(PILOT),
              'caveat': 'Unlisted hands are not confirmed ordinary play. AUC diagnoses retrieval of the supplied evidence, not the truth of strategic judgments.'}
    if args.full:
        result.update(full=discrimination(FULL), batch_shift=batch_shift(), input_scope=input_scope())
    (OUT / ('full_report.json' if args.full else 'pilot_report.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v['judgments'] for k, v in result['pilot'].items()}, indent=2))
