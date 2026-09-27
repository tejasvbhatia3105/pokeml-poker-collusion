\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
from pathlib import Path
import numpy as np,polars as pl
from sklearn.metrics import roc_auc_score
from session142_family_expert_transfer import ROOT as SOURCE,FAMILIES

ROOT=Path('artifacts/evidence_session144_expert_support');C=pl.col

def tail(p,minimum):
    z=np.zeros(minimum+1);z[0]=1
    for v in p:
        old=z.copy();z[0]=old[0]*(1-v);z[1:minimum]=old[1:minimum]*(1-v)+old[:minimum-1]*v;z[minimum]=old[minimum]+old[minimum-1]*v
    return z[-1]

def main():
    ROOT.mkdir(exist_ok=True);d=pl.read_parquet(sorted(SOURCE.glob('T*.parquet')));mins=json.load(open(SOURCE/'audit.json'))['minimums'];rows=[]
    for (pid,expert),g in d.group_by('pair_id','expert'):
        ca=g.select('primary','secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];p=.5*(ca+hp);fold=int(g['fold'][0]);k=mins[str(fold)][expert]
        rows.append(dict(pair_id=pid,expert=expert,family=g['true_family'][0],fold=fold,n_hands=len(g),joint_expected_count=float(p.sum()),joint_tail=float(tail(p.sum(1),k)),raw_top5_confidence=float(np.sort(g['raw_score'].to_numpy())[-5:].mean()),conditioned_top5_confidence=float(np.sort(g['conditioned_score'].to_numpy())[-5:].mean())))
    q=pl.DataFrame(rows);q.write_parquet(ROOT/'expert_support.parquet');fields=['joint_tail','raw_top5_confidence','conditioned_top5_confidence'];report=[]
    for held in FAMILIES:
        available=q.filter(C('expert')!=held);z=available.group_by('pair_id').agg(C('family').first(),C('fold').first(),*[C(n).max().alias(n) for n in fields]);known=(z['family']!=held).to_numpy();r=dict(withheld_family=held,known_pairs=int(known.sum()),unseen_pairs=int((~known).sum()),statistics={})
        for n in fields:
            v=z[n].to_numpy();r['statistics'][n]=dict(AUROC_known_vs_unseen=roc_auc_score(known,v),known_quantiles=np.quantile(v[known],[0,.1,.5,.9,1]).tolist(),unseen_quantiles=np.quantile(v[~known],[0,.1,.5,.9,1]).tolist())
        report.append(r)
    out=dict(method=__doc__,results=report,limitations='No fitted threshold, nested calibration, routing model or evaluation inference. Confidence summaries may differ by exposure and family; these are descriptive AUCs on reused labels, not a deployable unknown detector.')
    (ROOT/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
