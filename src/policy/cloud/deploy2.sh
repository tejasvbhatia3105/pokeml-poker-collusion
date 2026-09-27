#!/bin/zsh
SC=cache
L=$SC/cloud/deploy2.log; H=root@64.247.201.61; P=12447; K=$HOME/.ssh/id_ed25519
echo "== upload memmaps $(date +%H:%M:%S)" > $L
rsync -a -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $K -p $P" $SC/seq_tokens/*_X.npy $H:cache/seq_tokens/ >> $L 2>&1 || { echo UPLOAD_FAILED >> $L; exit 1; }
rsync -a -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $K -p $P" $SC/action_tokens/*_XA.npy $H:cache/action_tokens/ >> $L 2>&1 || { echo UPLOAD_FAILED >> $L; exit 1; }
echo "== smoke $(date +%H:%M:%S)" >> $L
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $K -p $P $H 'cd /workspace/pokeml && rm -rf artifacts/seq_v8_smoke/*.csv artifacts/seq_v8_smoke/*.pt && POLARS_MAX_THREADS=8 python3 src/policy/seq_train.py cache/smoke_tokens artifacts/seq_v8_smoke artifacts/seq_v8_smoke/config.json 2>&1 | grep -E "^device|train seqs|epoch|scored|^done|Traceback|Error" | tail -8' >> $L 2>&1
grep -q "^done" $L || { echo SMOKE_FAILED >> $L; exit 1; }
echo "== launch v8 $(date +%H:%M:%S)" >> $L
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $K -p $P $H 'cd /workspace/pokeml && rm -rf artifacts/seq_v8_smoke && nohup bash /workspace/pod_run_v8.sh > /workspace/run.out 2>&1 &' >> $L 2>&1
echo "== launched $(date +%H:%M:%S)" >> $L
