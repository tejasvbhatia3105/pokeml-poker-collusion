\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib
from pathlib import Path
import polars as pl
import session193_selected_membership_training as training
import session194_selected_map_training as ranking
from session189_pair_event_prototypes import data
from session195_restore_grounded import ROOT

def design():
    q=pl.read_parquet(ROOT/'grounded.parquet')
    d=data().join(q,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    cols=json.load(open(ROOT/'columns.json'))
    assert d.select(cols).null_count().to_numpy().sum()==0
    return d,cols

def control():
    return pl.read_parquet('artifacts/evidence_session62_grounded_list_boost/conditional_boost_oof.parquet').select('pair_id','hand_id',pl.col('full').alias('compact_control'))

def main():
    checks=json.load(open(ROOT/'restoration_verification.json'))
    assert len(checks['replayed_models'])==2 and all(r['prediction_error']==0 for r in checks['replayed_models'])
    cfg=dict(training.CONFIG,method=__doc__,feature_count=169,matched_control='full62 exact ordered-list NLL; equal pressure control is R33',
             feature_sha256=hashlib.sha256((ROOT/'grounded.parquet').read_bytes()).hexdigest())
    membership=training.objective
    for kind,fn,mathroot in [('membership',membership,Path('artifacts/evidence_session193_selection_marginals')),
                             ('smoothmap',ranking.objective,ranking.ROOT)]:
        dest=ROOT/kind;dest.mkdir(exist_ok=True)
        assert not list(dest.glob('fold*.joblib')),'Do not overwrite completed experiments'
        (dest/'math_verification.json').write_bytes((mathroot/'math_verification.json').read_bytes())
        training.ROOT=dest;training.objective=fn
        training.CONFIG=dict(cfg,loss_arm=kind,objective='selected Bernoulli NLL' if kind=='membership' else 'one minus smooth AP5',
                             rank_temperature=None if kind=='membership' else .25,cut_temperature=None if kind=='membership' else .5)
        training.main(design=design,control=control)
        p=dest/'report.json';r=json.loads(p.read_text());r['method']=__doc__;r['loss_arm']=kind
        r['arm_names']={'compact_control':'full62','action_list':'195_'+kind,'compact_pressure':'R33','action_pressure':'195_'+kind+'_with_pressure59'}
        p.write_text(json.dumps(r,indent=2))

if __name__=='__main__':main()
