#!/bin/zsh
SC=cache
L=$SC/cloud/seeds.log; H=root@64.247.201.61; P=12447; K=$HOME/.ssh/id_ed25519; PY=.venv/bin/python
SSHC="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $K -p $P"
cd .
echo "== rebuild tokens2/3 $(date +%H:%M:%S)" > $L
[ -d $SC/hand_rows2 ] || HAND_ROWS_ALL=$SC/hand_rows2 $PY src/policy/build_pair_features.py >> $L 2>&1
[ -f $SC/seq_tokens2/columns2.json ] || $PY src/policy/build_seq_tokens2.py $SC/seq_tokens $SC/seq_tokens2 >> $L 2>&1
[ -f $SC/seq_tokens3/columns3.json ] || $PY src/policy/build_seq_tokens3.py $SC/seq_tokens $SC/hand_rows2 artifacts/candidate_r4s/pair_oof_allpairs.csv artifacts/candidate_r12/eval_r4s_all.csv $SC/seq_tokens3 >> $L 2>&1
echo "== upload $(date +%H:%M:%S)" >> $L
rsync -a -e "$SSHC" $SC/seq_tokens2 $SC/seq_tokens3 $SC/cloud/pod_run_variant.sh $H:cache/ >> $L 2>&1 || { echo UPLOAD_FAILED >> $L; exit 1; }
for v in seq_v6 seq_v7; do sed "s|$SC/|cache/|g" artifacts/$v/config.json > $SC/cloud/${v}_config.json; done
rsync -a -e "$SSHC" $SC/cloud/seq_v6_config.json $SC/cloud/seq_v7_config.json $H:cache/ >> $L 2>&1
echo "== launch seeds $(date +%H:%M:%S)" >> $L
$SSHC $H 'cd /workspace/pokeml && chmod +x cache/pod_run_variant.sh && nohup bash -c "cache/pod_run_variant.sh seq_v6s1 cache/seq_v6_config.json 1; cache/pod_run_variant.sh seq_v6s2 cache/seq_v6_config.json 2" > /workspace/v6seeds.out 2>&1 & nohup bash -c "cache/pod_run_variant.sh seq_v7s1 cache/seq_v7_config.json 1; cache/pod_run_variant.sh seq_v7s2 cache/seq_v7_config.json 2" > /workspace/v7seeds.out 2>&1 & sleep 1; echo launched' >> $L 2>&1
echo "== SEEDS_LAUNCHED $(date +%H:%M:%S)" >> $L
