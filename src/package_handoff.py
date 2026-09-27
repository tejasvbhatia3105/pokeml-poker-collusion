from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/research_history'
out.mkdir(parents=True, exist_ok=True)
selected = [root / name for name in ['README.md', 'HANDOFF.md', 'RESEARCH_HISTORY.md']]
selected += [p for p in (root / 'src').rglob('*') if p.is_file()
             and p.suffix in {'.py', '.cpp', '.h', '.hpp', '.sh'} and '__pycache__' not in p.parts]
selected += list(root.glob('requirements*.txt'))
selected += [p for p in (root / 'artifacts').rglob('*') if p.is_file()
             and 'research_history' not in p.parts and (p.suffix == '.md'
             or (p.suffix == '.json' and p.stat().st_size < 2_000_000)
             or (p.suffix == '.csv' and p.stat().st_size < 1_000_000 and any(
                 s in p.name for s in ['metric', 'validation', 'audit', 'bootstrap', 'selection'])))]
selected += list((root / 'artifacts/reference_metric').glob('*.py'))
selected = sorted(set(selected))
manifest = []
for path in selected:
    contents = path.read_bytes()
    manifest.append({'path': str(path.relative_to(root)), 'bytes': len(contents),
                     'sha256': hashlib.sha256(contents).hexdigest()})
(out / 'manifest.json').write_text(json.dumps({
    'start_here': 'HANDOFF.md',
    'description': 'Research handoff, not a Kaggle upload or complete training environment. '
                   'Excludes data, large caches, models and submission files. '
                   'Current handoff/history supersede stale historical status text.',
    'files': manifest}, indent=2) + '\n')
archive = out / 'poker_research_history.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in selected:
        bundle.write(path, path.relative_to(root))
    bundle.write(out / 'manifest.json', 'manifest.json')
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for item in manifest:
        assert hashlib.sha256(bundle.read(item['path'])).hexdigest() == item['sha256']
print(json.dumps({'archive': str(archive), 'files': len(selected),
                  'bytes': archive.stat().st_size, 'manifest_verified': True}, indent=2))
