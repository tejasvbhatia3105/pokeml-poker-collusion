from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    'artifacts/v1/submission.csv': '859c71809fe5f30a1e3883cc4d984018ff78c1c4c93d4e18ce16056bf55f8a91',
    'artifacts/v4/submission.csv': 'b02a199de444c03fe9a2412ec4fcd4f63edea97b5370f8a283dc15f54fba98c1',
    'artifacts/v5/submission.csv': '8489232a45c0646e7dfd24c48ea93aa654091dfdc03e16f63863d467c5fc1a28',
    'artifacts/diagnostics/v4_pair_ranking/submission.csv': 'fd47b52bc2311b6b14d1908950b7a7146c5b34c100e5d5220610ab8bc2b3db97',
}
required = ['data/' + f for f in [
    'players.parquet', 'hands.parquet', 'seats.parquet', 'actions.parquet',
    'development_labels.csv', 'development_evidence.csv', 'evaluation_pairs.csv', 'sample_submission.csv']]
required += ['artifacts/folds.json', 'artifacts/reference_metric/reference_metric.py',
             'artifacts/policy/cards.dylib', 'artifacts/policy/residual_columns.json',
             'artifacts/policy/residual_oof.csv', 'artifacts/policy/residual_eval.csv',
             'artifacts/rank_columns.json', 'artifacts/policy/sequence/evidence_columns.json']
for fold in range(4):
    required += [f'artifacts/policy/{kind}_fold{fold}.cbm' for kind in ['action', 'size', 'residual']]
    for family in ['directed_transfer', 'soft_play', 'coordinated_isolation']:
        required += [f'artifacts/rank_{family}_fold{fold}.cbm',
                     f'artifacts/policy/sequence/evidence_{family}_fold{fold}.cbm']
missing = [p for p in required if not (ROOT / p).is_file()]
hash_status = {}
for name, expected in HASHES.items():
    path = ROOT / name
    if not path.is_file():
        hash_status[name] = 'missing'
    else:
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        hash_status[name] = 'match' if actual == expected else 'MISMATCH: ' + actual
counts = {name: len(list((ROOT / 'artifacts/policy' / name).glob('*.parquet')))
          for name in ['actions', 'states', 'hand_features', 'pair_features']}
print(json.dumps({'root': str(ROOT), 'python': sys.version, 'missing_assets': missing,
                  'archived_submission_hashes': hash_status, 'policy_partition_counts': counts,
                  'active_experiment_environment': {k: os.environ[k] for k in
                      ['POLICY_WINDOW', 'ONE_TABLE', 'ONLY_STATES', 'RELATIONSHIP_CONTEXT'] if k in os.environ},
                  'scope': 'Asset presence and archive integrity only; no training, model loading, cache freshness proof or upload.'}, indent=2))
sys.exit(1 if missing or any(v != 'match' for v in hash_status.values()) else 0)
