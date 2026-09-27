# Poker collusion detection: reproduction repository

First-place solution for the poker collusion detection competition. Selected submissions: `r104` (private 0.94320, public 0.94054) and `r103` (private 0.94311). The write-up is in `docs/WRITEUP_r104.md`.

Everything infers coordination from gameplay (betting, cards, chip flow). Player, hand and pair identifiers are used only to join tables and to define held-out pools.

## Layout

```
src/                      pipeline code (comment-free by convention)
src/policy/               policy models, pair rankers, sequence models, evidence stack
src/policy/session3/      R12 / R23 / R26 blend builders
src/policy/session4r/     fourth-family probe, evidence-strength lifts, final evidence lists (R33 -> r104)
src/policy/cloud/         helper scripts used on rented GPU pods
configs/                  training configurations copied from the archived runs
artifacts/folds.json      fixed player-pool folds (4 folds over tables, StratifiedGroupKFold seed 42)
requirements.txt          pinned Python dependencies (Python 3.14)
docs/WRITEUP_r104.md      solution write-up
```

All scripts are run from the repository root. Paths inside the scripts are relative (`data/...`, `artifacts/...`). Every script that skips tables whose output already exists does so silently: delete the output directory to force a rebuild.

## Setup

```
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch==2.8.0          # sequence models only; not in requirements.txt
mkdir -p data artifacts/policy
```

Put the eight competition files in `data/`: `hands.parquet`, `seats.parquet`, `actions.parquet`, `players.parquet`, `development_labels.csv`, `development_evidence.csv`, `evaluation_pairs.csv`, `sample_submission.csv`.

Compile the native card evaluators (the Python wrappers load them by these file names on every platform):

```
c++ -O3 -std=c++17 -shared -fPIC -o artifacts/policy/cards.dylib src/policy/cards.cpp
mkdir -p artifacts/evidence_session5 artifacts/evidence_session50_matchup
c++ -O3 -std=c++17 -shared -fPIC -o artifacts/evidence_session5/multiway.dylib src/policy/session5_multiway.cpp
c++ -O3 -std=c++17 -shared -fPIC -o artifacts/evidence_session50_matchup/matchup.dylib src/policy/session50_matchup.cpp
PYTHONPATH=src/policy .venv/bin/python src/policy/test_cards.py
```

Environment used for the selected submission: macOS laptop (Apple M-series, 16 GB) for everything except the transformer training, which ran on rented GPU pods (H100, then RTX 5090) with the same code. Thread caps inside the scripts (`POLARS_MAX_THREADS`, CatBoost `thread_count`) are set for that laptop.

Set `PY=.venv/bin/python` and `SC=cache` (any writable directory for the large token caches) in the commands below.

## Stage 0: partitions

```
$PY src/prepare_compact.py
```

Writes `artifacts/compact/{hands,seats,actions}/table_id=*/`. About 25 seconds.

## Stage 1: policy caches and the action / bet-size models

```
PYTHONPATH=src/policy $PY src/policy/prepare_actions.py    # artifacts/policy/{actions,states}/{table}.parquet, feature_columns.json
$PY src/policy/train_ordinary.py                            # action_fold{0..3}.cbm, size_fold{0..3}.cbm, table_folds.json
```

Cross-fitted CatBoost action (4-class) and bet-size models over the four pool folds. About 5 minutes.

## Stage 2: per-hand channels, pair features, residual models

```
$PY src/policy/build_pair_features.py                                   # artifacts/policy/pair_features, hand_features (full history)
for w in first_2000 last_2000 w500_2500 w0_1500 w750_2250 w1500_3000; do
  POLICY_WINDOW=$w $PY src/policy/build_pair_features.py               # artifacts/policy/full_window_stress/$w/...
done
HAND_ROWS_ALL=$SC/hand_rows2 $PY src/policy/build_pair_features.py     # per-hand channel rows for every pair -> $SC/hand_rows2
POKER_PARTITION_ROOT=artifacts/compact $PY src/context_pairs.py                 # artifacts/context_pairs
$PY src/policy/evaluate_pairs.py                                        # residual_fold{0..3}.cbm, residual_columns.json, residual_oof.csv
```

## Stage 3: CatBoost pair rankers (r4s, r4k), leave-partner-out (R12)

```
for c in r4s r4k; do
  mkdir -p artifacts/candidate_$c && cp configs/candidate_$c.json artifacts/candidate_$c/config.json
  $PY src/policy/train_window_augmented.py artifacts/candidate_$c artifacts/candidate_$c/config.json
done
mkdir -p artifacts/candidate_r12
$PY src/policy/leave_partner_out.py artifacts/candidate_r4s evaluation $SC/hand_rows2 artifacts/candidate_r12/eval_r4s_all.csv --mode all
$PY src/policy/leave_partner_out.py artifacts/candidate_r4k evaluation $SC/hand_rows2 artifacts/candidate_r12/eval_r4k_all.csv --mode all
$PY src/policy/session3/build_candidate_r12.py artifacts/candidate_r12    # pair_eval.csv = geometric mean of the two rescored rankers
```

`r4s` uses the first and last 2,000-hand crops, `r4k` six crops; both use three seeds, pseudo-positives above 0.9, 12,000 background pairs at weight 0.1, 900 iterations, depth 5 (see `configs/`). Leave-partner-out defaults are `--policy min --all-thr 0.3 --min-risk 0.02`.

## Stage 4: sequence token caches

```
$PY src/policy/build_seq_tokens.py  $SC/hand_rows2 $SC/seq_tokens
$PY src/policy/build_seq_tokens2.py $SC/seq_tokens $SC/seq_tokens2
$PY src/policy/build_seq_tokens3.py $SC/seq_tokens $SC/hand_rows2 artifacts/candidate_r4s/pair_oof_allpairs.csv artifacts/candidate_r12/eval_r4s_all.csv $SC/seq_tokens3
$PY src/policy/build_action_tokens.py $SC/seq_tokens $SC/action_tokens
$PY src/policy/build_hand_states.py  $SC/seq_tokens $SC/hand_states
```

About 10 minutes. The trainer creates `{table}_X.npy` and `{table}_XA.npy` memory maps next to the archives on first use; raise the open-file limit (`ulimit -n 65536`) before training.

## Stage 5: sequence models (GPU)

The configs in `configs/seq_v6.json`, `configs/seq_v7.json`, `configs/seq_v8.json` reference the cache directories as `$SC/...`; substitute the real path (`sed "s|\$SC|$SC|g"`). Seeds 1 and 2 reuse the base config.

```
for spec in "seq_v6 0" "seq_v6s1 1" "seq_v6s2 2" "seq_v7 0" "seq_v7s1 1" "seq_v7s2 2" "seq_v8 0"; do
  set -- $spec; V=artifacts/$1; base=${1%s[12]}; mkdir -p $V
  sed "s|\$SC|$SC|g" configs/$base.json > $V/config.json
  SEED=$2 $PY src/policy/seq_train.py $SC/seq_tokens $V $V/config.json
  $PY src/policy/seq_rescore.py $SC/seq_tokens $V development $V/dev_full.csv $V/dev_full_lpo.csv
  $PY src/policy/seq_rescore.py $SC/seq_tokens $V development $V/dev_first_2000.csv $V/dev_first_2000_lpo.csv --window first_2000
  $PY src/policy/seq_rescore.py $SC/seq_tokens $V development $V/dev_last_2000.csv $V/dev_last_2000_lpo.csv --window last_2000
  $PY src/policy/seq_rescore.py $SC/seq_tokens $V evaluation $V/eval_all.csv $V/eval_all_lpo.csv
done
```

`seq_train.py` picks CUDA when available, otherwise Apple MPS, otherwise CPU. Measured on one H100: v6 about 15 minutes, v7 about 30 minutes, v8 about an hour per model including the four rescoring passes. `src/policy/cloud/pod_run_variant.sh NAME CONFIG SEED` wraps the same five commands for a pod.

## Stage 6: R23 and the R26 blend

```
$PY src/policy/session3/build_candidate_r23.py artifacts/candidate_r12/pair_eval.csv artifacts/seq_v6/eval_all_lpo.csv artifacts/candidate_r23
$PY src/policy/session3/build_candidate_pair_eval.py artifacts/candidate_r26        # defaults: --v6 seq_v6,seq_v6s1,seq_v6s2 --v7 seq_v7,seq_v7s1,seq_v7s2 --v8 seq_v8 --w8 0.5
$PY src/policy/assemble_candidate.py --pair-predictions artifacts/candidate_r26/pair_eval.csv --output artifacts/candidate_r26/submission.csv
```

Risk = sqrt(R12 risk x exp((mean log v6 + mean log v7 + 0.5 log v8) / 2.5)); family shares come from R23 (R12 x seq_v6, 1:1). `assemble_candidate.py` attaches the relationship-evidence lists that the later evidence stack replaces.

## Stage 7: evidence stack (R26 -> R33)

Per-family boosted hand rankers, an ordered-list model and several inference layers, built over sessions 4 to 65. Every builder asserts the SHA-256 of the submission it starts from, so the stages must be run in this order and each must reproduce its predecessor exactly. Training scripts have no arguments unless shown; a few take environment variables.

```
# session 4 evidence models and R27
$PY src/policy/session4_nested_evidence.py; $PY src/policy/session4_leafwise_evidence.py; $PY src/policy/session4_nested_blend.py
EVIDENCE_SET_ROOT=artifacts/evidence_session4/nested_blend $PY src/policy/session4_set_evidence.py
$PY src/policy/session4_build_candidate.py                                   # artifacts/candidate_r27
# R28
PRIORITY_ORDERED=1 $PY src/policy/session6_priority.py; $PY src/policy/session6_compare.py; $PY src/policy/session6_order_audit.py
$PY src/policy/session6_build_candidate.py                                   # artifacts/candidate_r28
# R29
$PY src/policy/session7_hist_events.py; $PY src/policy/session7_calibration.py; $PY src/policy/session7_compare.py
$PY src/policy/session7_build_candidate.py                                   # artifacts/candidate_r29
# R30
$PY src/policy/session9_nested_inputs.py; $PY src/policy/session10_nested6.py
LIST_INPUT_ROOT=artifacts/evidence_session10/nested6 LIST_OUTPUT_ROOT=artifacts/evidence_session10/nested6_model LIST_KINDS=independent $PY src/policy/session10_list_learning.py
$PY src/policy/session11_conditional_family.py; BOOST_KINDS=compact,full $PY src/policy/session11_list_boost.py
EVAL_CACHE_OUTPUT=artifacts/evidence_session11/eval_cache $PY src/policy/session11_eval_cache.py
EVAL_CACHE_OUTPUT=artifacts/evidence_session11/tail_cache EVAL_CACHE_MIN=.01 EVAL_CACHE_MAX=.05 $PY src/policy/session11_eval_cache.py
CANDIDATE_METHOD=conditional_family $PY src/policy/session11_build_candidate.py   # artifacts/candidate_r30
# R31: event heads and paired inference (sessions 5, 12, 17, 25, 26, 27, 29, 31, 35 train the inputs these read)
$PY src/policy/session36_nested_cascade.py; $PY src/policy/session36_verify_inputs.py
$PY src/policy/session37_bet_fold.py; $PY src/policy/session38_soft_bet_fold.py; $PY src/policy/session41_isolation_bet_fold.py
$PY src/policy/session46_verify.py; $PY src/policy/session46_build_candidate.py    # artifacts/candidate_r31
# R32: matchup features
$PY src/policy/session50_matchup.py; $PY src/policy/session50_ablation.py; $PY src/policy/session50_verify.py
$PY src/policy/session50_ablation_verify.py; $PY src/policy/session50_representation_audit.py
$PY src/policy/session51_verify.py; $PY src/policy/session51_build_candidate.py    # artifacts/candidate_r32
# R33: pressure and grounded list models
$PY src/policy/session55_current_targets.py; $PY src/policy/session55_current_nested.py --pilot; $PY src/policy/session55_current_nested.py; $PY src/policy/session55_verify.py
LIST_INPUT_ROOT=artifacts/evidence_session55_current_nested LIST_OUTPUT_ROOT=artifacts/evidence_session55_current_nested/correction $PY src/policy/session11_conditional_family.py
$PY src/policy/session57_isolation_pressure.py; $PY src/policy/session59_pressure_equity.py
$PY src/policy/session62_grounded_list_boost.py; $PY src/policy/session62_compare.py; $PY src/policy/session64_equal_list_pressure.py
$PY src/policy/session63_verify.py; $PY src/policy/session65_verify.py
$PY src/policy/session65_build_candidate.py                                  # artifacts/candidate_r33 (+ inference/T*.parquet per table)
```

Sessions not named above (`session5_*`, `session12_*`, `session17_*`, `session25_*`, `session26_*`, `session27_*`, `session29_*`, `session31_*`, `session35_*`, `session39_*`, `session42_*`, `session48_*`, `session58_*`) train inputs that the listed scripts load; run them first, in numeric order. Each script states its inputs at the top of the file.

## Stage 8: fourth family, evidence strength, final lists (R33 -> r104)

```
POKEML_SCRATCH=$SC/session4r PY=$PY bash src/policy/session4r/run_final_layers.sh
```

The runner executes, in order: the card-sharing probe over every ordered pair; the fourth-family membership sets (`candidate_r34` members, `candidate_r41` new pairs, `lift_candidates2`); the pump-restricted evidence lists (`build_pump_evidence_variant.py`, r42 to r58); the planted-versus-natural raise classifier; the evidence-strength lifts (r70, r72); the `other_coordination` relabel with the 16 confident members kept (r75, r77, r80); the zone lifts (r88); the primary-plus-fallback lists (r92); and the reverse-direction fallback (r95, r103, r104). Each step is one script with the exact thresholds used. Inputs beyond `candidate_r33`: `candidate_r26/pair_eval.csv` and `candidate_r30/evidence_scores.parquet`.

Outputs: `artifacts/candidate_r104_revk5/submission.csv` (selected) and `artifacts/candidate_r103_final_inter/submission.csv` (`--revk 8` instead of `--revk 5`).

Validate any submission with:

```
$PY src/validate_submission.py artifacts/candidate_r104_revk5/submission.csv --report artifacts/candidate_r104_revk5/validation.json
```

## Archived models

`models/` holds the weights behind the selected submission, for inspection and for re-running the later stages without retraining the earlier ones:

```
models/policy/            action_fold{0..3}.cbm, size_fold{0..3}.cbm, residual_fold{0..3}.cbm   (stages 1 and 2)
models/candidate_r4s/     pair_seed{991,1991,2991}_fold{0..3}.cbm + config.json                  (stage 3)
models/candidate_r4k/     same, six-crop configuration
models/seq_v6*, seq_v7*, seq_v8/   seq_fold{0..3}.pt, norm.npz, config.json                      (stage 5)
```

Copy a directory over the matching `artifacts/...` directory to reuse it. The column-order files next to them (`feature_columns.json`, `residual_columns.json`, `table_folds.json`) are regenerated identically by stages 1 and 2. `seq_train.py` retrains a fold unless that fold's checkpoint and its scored CSV files are both present, so the sequence weights alone do not skip training.

## Verification status

Measured on 26 and 27 September 2026 against the files that were uploaded to the competition.

- **Stages 0 to 3 re-run from the raw data** (laptop, 30 minutes). The fresh R12 ranking agrees with the archived one at correlation 0.9991 and shares 496 of its top 500 pairs; the residual difference is CatBoost thread nondeterminism, not a code difference.
- **Blend builders replay exactly.** From the archived per-model outputs, `build_candidate_r12.py` and `build_candidate_r23.py` reproduce the archived R12 and R23 tables to 0.0, and `build_candidate_pair_eval.py` reproduces the archived R26 risks to 1.7e-16.
- **Stage 8 runs end to end** from the archived `candidate_r33`, `candidate_r26/pair_eval.csv` and `candidate_r30/evidence_scores.parquet` on the fresh policy caches (25 steps, 4 minutes). Its first step, `candidate_r34`, reproduces the original file hash exactly, so the caches are deterministic. Its final output agrees with the uploaded r104 on 112,523 of 112,540 rows; 10 risk values and 7 evidence lists differ. Cause: when the uploaded file was built, some of R33's per-table inference files were unreadable on the build machine (cloud-evicted) and the scripts skipped them (53 of 65 fourth-family members had hand scores then, 65 of 65 now; 58 of 77 then, 71 of 77 now; the evidence-strength lift set had 14 pairs then, 18 now). The scripted run uses every file. The uploaded file is therefore a slightly under-informed instance of this recipe, not a different recipe.
- **Stage 5 (sequence models) re-run from the fresh token caches** on a rented A100 (seven models, 2.3 hours). Blending the fresh transformer risks with the archived R12 and R23 through `build_candidate_pair_eval.py` gives a ranking at Pearson 0.9987 and Spearman 0.983 with the archived R26, sharing 494 of its top 500 pairs. The fully fresh chain (fresh R12 and fresh transformers) gives 0.9985, 0.982 and 494 of 500. Individual models agree with their archived counterparts at 0.95 (v6), 0.98 (v7) and 0.98 (v8); the blend averages the seed-to-seed variation out.
- **Stage 7 (evidence stack, R26 to R33)** was not re-run after the competition. Its scripts are included unchanged with the archived SHA gates, and the archived R33 outputs were used for the stage-8 verification above.
