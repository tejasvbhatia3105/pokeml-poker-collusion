#!/bin/bash
set -e
mkdir -p /workspace && cd /workspace
tar -xf /workspace/pokeml_cloud.tar && cd /workspace/pokeml
pip install -q polars numpy 2>&1 | tail -1
SC=cache
sed -i "s|dev=torch.device('mps' if torch.backends.mps.is_available() and cfg.get('mps',True) else 'cpu')|dev=torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() and cfg.get('mps',True) else 'cpu'))|" src/policy/seq_train.py
grep -c "cuda" src/policy/seq_train.py
sed -i "s|$SC/|cache/|g" artifacts/seq_v8/config.json; cat artifacts/seq_v8/config.json | tr -d '\n' | cut -c1-400; echo
mkdir -p cache/smoke_tokens artifacts/seq_v8_smoke; cp cache/seq_tokens/columns.json cache/smoke_tokens/
for t in $(ls cache/seq_tokens/*.npz | head -24); do ln -sf /workspace/pokeml/$t cache/smoke_tokens/$(basename $t); done
python3 - <<'PY'
import json; c=json.load(open('artifacts/seq_v8/config.json')); c.update(epochs=1,hardneg=300,background=600); json.dump(c,open('artifacts/seq_v8_smoke/config.json','w'))
PY
POLARS_MAX_THREADS=8 python3 src/policy/seq_train.py cache/smoke_tokens artifacts/seq_v8_smoke artifacts/seq_v8_smoke/config.json 2>&1 | grep -v Warning | grep -E "device|train seqs|epoch|scored|done|Traceback|Error" | tail -8
echo SMOKE_OK
