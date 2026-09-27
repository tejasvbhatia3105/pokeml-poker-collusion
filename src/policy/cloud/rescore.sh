#!/bin/zsh
SC=cache
L=$SC/cloud/rescore.log; H=root@64.247.201.61; P=12447; K=$HOME/.ssh/id_ed25519
cd .
echo "== upload states $(date +%H:%M:%S)" > $L
rsync -a -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $K -p $P" artifacts/policy/states $H:/workspace/pokeml/artifacts/policy/ >> $L 2>&1 || { echo UPLOAD_FAILED >> $L; exit 1; }
echo "== rescore $(date +%H:%M:%S)" >> $L
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $K -p $P $H 'cd /workspace/pokeml && V=artifacts/seq_v8 && export POLARS_MAX_THREADS=8 && for a in "development dev_full full" "development dev_first_2000 first_2000" "development dev_last_2000 last_2000" "evaluation eval_all full"; do set -- $a; w=""; [ "$3" != full ] && w="--window $3"; python3 src/policy/seq_rescore.py cache/seq_tokens $V $1 $V/$2.csv $V/${2}_lpo.csv $w 2>&1 | grep -E "candidates|rescored|Traceback|Error" | tail -3; done; ls $V/*_lpo.csv | wc -l' >> $L 2>&1
echo "== pull $(date +%H:%M:%S)" >> $L
mkdir -p artifacts/seq_v8
rsync -a -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $K -p $P" --exclude _tmp $H:/workspace/pokeml/artifacts/seq_v8/ artifacts/seq_v8/ >> $L 2>&1
echo "== RESCORE_DONE $(date +%H:%M:%S)" >> $L
