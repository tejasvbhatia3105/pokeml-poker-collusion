#!/bin/zsh
cd .
SC=cache
mkdir -p $SC/cloud/pokeml/{src/policy,artifacts/candidate_r4s,artifacts/seq_v8,data,cache}
cp src/policy/seq_train.py src/policy/seq_rescore.py src/policy/eval_window_csv.py $SC/cloud/pokeml/src/policy/
cp artifacts/folds.json $SC/cloud/pokeml/artifacts/; cp artifacts/candidate_r4s/pair_oof_allpairs.csv $SC/cloud/pokeml/artifacts/candidate_r4s/
cp artifacts/seq_v8/config.json $SC/cloud/pokeml/artifacts/seq_v8/
cp data/development_labels.csv data/development_evidence.csv $SC/cloud/pokeml/data/
ln -sfn $SC/seq_tokens $SC/cloud/pokeml/cache/seq_tokens; ln -sfn $SC/action_tokens $SC/cloud/pokeml/cache/action_tokens; ln -sfn $SC/hand_states $SC/cloud/pokeml/cache/hand_states
cd $SC/cloud && COPYFILE_DISABLE=1 tar --exclude='*_X.npy' --exclude='*_XA.npy' -chf pokeml_cloud.tar pokeml && ls -la pokeml_cloud.tar | awk '{print $5}' > pack.done
