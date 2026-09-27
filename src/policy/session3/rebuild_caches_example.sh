#!/bin/zsh
set -e
cd .
SC=cache
PY=.venv/bin/python; L=$SC/rebuild.log
step(){ echo "== $1 $(date +%H:%M:%S)" >> $L; }
step hand_rows2; HAND_ROWS_ALL=$SC/hand_rows2 $PY src/policy/build_pair_features.py >> $L 2>&1
step seq_tokens; $PY src/policy/build_seq_tokens.py $SC/hand_rows2 $SC/seq_tokens >> $L 2>&1
step seq_tokens2; $PY src/policy/build_seq_tokens2.py $SC/seq_tokens $SC/seq_tokens2 >> $L 2>&1
step seq_tokens3; $PY src/policy/build_seq_tokens3.py $SC/seq_tokens $SC/hand_rows2 artifacts/candidate_r4s/pair_oof_allpairs.csv artifacts/candidate_r12/eval_r4s_all.csv $SC/seq_tokens3 >> $L 2>&1
step action_tokens; $PY src/policy/build_action_tokens.py $SC/seq_tokens $SC/action_tokens >> $L 2>&1
step allpairs; $PY - <<'PY' >> $L 2>&1
import polars as pl
S='cache/'
lab=pl.read_csv('data/development_labels.csv')
key=pl.concat([lab.select('player_1','player_2','label'),lab.select(pl.col('player_2').alias('player_1'),pl.col('player_1').alias('player_2'),'label')]).unique(['player_1','player_2'])
r=pl.read_csv('artifacts/candidate_r4s/pair_oof_allpairs.csv').select('pair_id',(1-pl.col('none')).alias('risk'))
for w in ['full','first_2000','last_2000']:
    d=pl.read_csv(f'artifacts/seq_v6/dev_{w}.csv').select('pair_id','player_1','player_2').join(key,on=['player_1','player_2'],how='left').with_columns(pl.col('label').fill_null(0)).join(r,on='pair_id',how='left')
    d.write_parquet(S+f'allpairs_{w}.parquet'); print(w,d.height,int(d['label'].sum()))
PY
step train; $PY src/policy/seq_train.py $SC/seq_tokens artifacts/seq_v7 artifacts/seq_v7/config.json >> artifacts/seq_v7/train.log 2>&1
step rescore; V=artifacts/seq_v7
$PY src/policy/seq_rescore.py $SC/seq_tokens $V development $V/dev_full.csv $V/dev_full_lpo.csv >> $V/train.log 2>&1
$PY src/policy/seq_rescore.py $SC/seq_tokens $V development $V/dev_first_2000.csv $V/dev_first_2000_lpo.csv --window first_2000 >> $V/train.log 2>&1
$PY src/policy/seq_rescore.py $SC/seq_tokens $V development $V/dev_last_2000.csv $V/dev_last_2000_lpo.csv --window last_2000 >> $V/train.log 2>&1
$PY src/policy/seq_rescore.py $SC/seq_tokens $V evaluation $V/eval_all.csv $V/eval_all_lpo.csv >> $V/train.log 2>&1
step eval; $PY src/policy/eval_window_csv.py $V/dev_full.csv:full $V/dev_full_lpo.csv:full $V/dev_first_2000_lpo.csv:first_2000 $V/dev_last_2000_lpo.csv:last_2000 > $V/dev_eval.txt 2>&1
echo SEQV7_DONE >> $V/dev_eval.txt; step done
