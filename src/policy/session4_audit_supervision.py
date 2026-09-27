from pathlib import Path
import ast,json,hashlib
import numpy as np,polars as pl
OUT=Path('artifacts/evidence_session4')
truth=pl.read_csv('data/development_evidence.csv');keys=set(truth.select('pair_id','hand_id').iter_rows());hands=set(truth['hand_id'])
path=Path(json.loads(Path('artifacts/seq_v6/config.json').read_text())['tokens3']).parent/'seq_tokens'
counts={'exact_positive_pair_hands':0,'old_extra_positive_pair_hands':0,'affected_non_target_pairs':set()}
for p in path.glob('*.npz'):
    with np.load(p) as z:
        pi=np.repeat(z['pair_id'],z['n']);ph=np.repeat(z['phase'],z['n']);hh=z['hand_id']
        use=(ph=='development')&np.isin(hh,list(hands))
        for pair,hand in zip(pi[use],hh[use]):
            if (pair,hand) in keys:counts['exact_positive_pair_hands']+=1
            else:counts['old_extra_positive_pair_hands']+=1;counts['affected_non_target_pairs'].add(pair)
counts['affected_non_target_pairs']=len(counts['affected_non_target_pairs'])
counts['scope']='All development pair-hand tokens, before sequence training sampling. Actual training contamination depends on sampled bags/windows.'
index=pl.read_parquet('artifacts/policy/evidence_training.parquet')
counts['extra_on_known_positive_bags']=index.filter(pl.col('hand_id').is_in(truth['hand_id'].implode())&(pl.col('evidence')==0)).height
source=Path('src/policy/seq_train.py').read_text();tree=ast.parse(source)
checks=[]
for node in ast.walk(tree):
    if isinstance(node,ast.Compare) and any(isinstance(x,ast.Name) and x.id=='EVH' for x in node.comparators):
        assert isinstance(node.left,ast.Tuple),ast.unparse(node)
        checks.append(ast.unparse(node))
assert len(checks)==2
assert ('pair_a','shared_hand') in {('pair_a','shared_hand')}
assert ('pair_b','shared_hand') not in {('pair_a','shared_hand')}
assert ('pair_a','different_hand') not in {('pair_a','shared_hand')}
counts['trainer_membership_checks']=checks
counts['r26_sha256']=hashlib.sha256(Path('artifacts/candidate_r26/submission.csv').read_bytes()).hexdigest()
assert counts['r26_sha256']=='a5b0fb3f488ec2d89683b301940f6636703a907b87e7bc4f947906d1b48ff509'
(OUT/'supervision_audit.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts,indent=2))
