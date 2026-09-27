import os,json,hashlib,ast,py_compile
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from evidence_data import load
from session5_pairwise_model import load_models,rankings
from session5_multiway_features import build
ROOT=Path('artifacts/evidence_session5');OLD=Path('artifacts/evidence_session4');P=Path('artifacts/policy');C=pl.col;report={}
expected={'r26':'a5b0fb3f488ec2d89683b301940f6636703a907b87e7bc4f947906d1b48ff509','r27':'1cef3d932b0fea2e669980ab8b4f367f89352b5aaf1553bb3a86b2546873f9ae'}
for n,h in expected.items():
    actual=hashlib.sha256(Path(f'artifacts/candidate_{n}/submission.csv').read_bytes()).hexdigest();assert actual==h;report[n+'_unchanged_sha256']=actual
nodes=[n for n in ast.walk(ast.parse(Path('src/policy/seq_train.py').read_text())) if isinstance(n,ast.Compare) and any(isinstance(c,ast.Name) and c.id=='EVH' for c in n.comparators)]
assert len(nodes)==2 and all(isinstance(n.left,ast.Tuple) and len(n.left.elts)==2 for n in nodes)
report['pair_hand_membership_sites']=len(nodes)
files=list(Path('src/policy').glob('session5_*.py'))+[Path('src/policy')/n for n in ['seq_contract.py','seq_train.py','seq_rescore.py','seq_hand_evidence.py']]
for path in files:py_compile.compile(str(path),doraise=True)
report['compiled_python_files']=len(files)
d,_=load();extra=pl.read_parquet(list((P/'outcome_roles').glob('T*.parquet'))).join(pl.read_parquet(list((P/'relationship_evidence').glob('T*.parquet'))),on=['pair_id','hand_id'],validate='1:1');d=d.join(extra,on=['pair_id','hand_id'])
ix=pl.read_parquet(OLD/'hand_index.parquet');d=d.join(ix.select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
models={backend:load_models(backend) for backend in ['cat','context']}
targets={backend:dict(pl.read_csv(ROOT/f'{prefix}_comparison.csv').select('pair_id','comparator_quarter').iter_rows()) for backend,prefix in [('cat','pairwise'),('context','pairwise_context')]};checked=0;maxerr=0.
with threadpool_limits(limits=4):
    for f in range(4):
        q=d.filter(C('fold')==f).join(pl.read_parquet(OLD/f'nested_blend/nested_outer{f}.parquet'),on=['pair_id','hand_id']);w=np.load(OLD/f'nested_blend/unary_weights_fold{f}.npy')
        for (pid,),g in q.group_by('pair_id'):
            for backend in ['cat','context']:
                hands=rankings(g,w,models[backend],[f],backend)['comparator_quarter'];true=set(g.filter(C('evidence')==1)['hand_id']);y=np.array([h in true for h in hands]);ap=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/min(5,len(true)))
                maxerr=max(maxerr,abs(ap-targets[backend][pid]));checked+=1
                if checked<=16:
                    assert rankings(g.reverse(),w,models[backend],[f],backend)['comparator_quarter']==hands
                    shuffled_labels=g.with_columns(C('evidence').reverse());assert rankings(shuffled_labels,w,models[backend],[f],backend)['comparator_quarter']==hands
assert checked==744 and maxerr<1e-12
report.update(comparator_inference_checked_pair_model_cases=checked,comparator_inference_max_map_difference=maxerr,candidate_order_invariance_passed=True,inference_ignores_evidence_labels=True)
table=ix['table_id'][0];query=ix.filter(C('table_id')==table).select('pair_id','hand_id').head(20).join(pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2'),on='pair_id')
a=build(table,query).sort('pair_id','hand_id');sw=query.rename({'player_1':'player_2','player_2':'player_1'});b=build(table,sw).sort('pair_id','hand_id')
assert a.select('pair_id','hand_id').equals(b.select('pair_id','hand_id'))
err=float(np.max(np.abs(a.select(C('^multi_.*$')).to_numpy()-b.select(C('^multi_.*$')).to_numpy())))
assert err<1e-5;report['multiway_endpoint_swap_max_difference']=err
(ROOT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
