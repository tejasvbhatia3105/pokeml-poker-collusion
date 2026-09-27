#!/bin/bash
cd /workspace/pokeml; V=artifacts/$1; mkdir -p $V; cp $2 $V/config.json; export POLARS_MAX_THREADS=6
SEED=$3 python3 src/policy/seq_train.py cache/seq_tokens $V $V/config.json >> $V/train.log 2>&1 || { echo TRAIN_FAILED >> $V/train.log; exit 1; }
python3 src/policy/seq_rescore.py cache/seq_tokens $V development $V/dev_full.csv $V/dev_full_lpo.csv >> $V/train.log 2>&1
python3 src/policy/seq_rescore.py cache/seq_tokens $V development $V/dev_first_2000.csv $V/dev_first_2000_lpo.csv --window first_2000 >> $V/train.log 2>&1
python3 src/policy/seq_rescore.py cache/seq_tokens $V development $V/dev_last_2000.csv $V/dev_last_2000_lpo.csv --window last_2000 >> $V/train.log 2>&1
python3 src/policy/seq_rescore.py cache/seq_tokens $V evaluation $V/eval_all.csv $V/eval_all_lpo.csv >> $V/train.log 2>&1
echo VARIANT_DONE >> $V/train.log
