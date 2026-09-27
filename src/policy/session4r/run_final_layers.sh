#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python}
D=src/policy/session4r
V=artifacts/candidate_r33/submission.csv
[ -d artifacts/policy/states ] && [ -n "$(ls artifacts/policy/states/*.parquet 2>/dev/null | head -1)" ] || { echo "rebuild artifacts/policy/{actions,states} from artifacts/compact"; PYTHONPATH=src/policy $PY src/policy/prepare_actions.py; }
echo "card_share_z.parquet"
$PY $D/card_sharing_probe.py
echo "card_share_eval.parquet, card_share_dev.parquet"
$PY $D/step10_card_share_eval.py
echo "artifacts/candidate_r34 (build_manifest members)"
$PY $D/build_r34_pump.py artifacts/candidate_r34
echo "artifacts/candidate_r37b"
$PY $D/build_r37_hybrid.py $V artifacts/candidate_r37b
echo "artifacts/candidate_r40"
$PY $D/step20_r40_floor.py
echo "artifacts/candidate_r41 (build_manifest new_pairs)"
$PY $D/step30_r41_lift.py
echo "pump_probe2.parquet"
$PY $D/pump_probe2.py
echo "pump_probe2_eval.parquet"
$PY $D/step40_pump_probe2_eval.py
echo "lift_candidates2.parquet"
$PY $D/step50_lift_candidates2.py
echo "artifacts/candidate_r42_chrono"
$PY $D/build_pump_evidence_variant.py artifacts/candidate_r41/submission.csv artifacts/candidate_r42_chrono --order r33 --chrono 8
echo "artifacts/candidate_r45_k8time"
$PY $D/build_pump_evidence_variant.py artifacts/candidate_r42_chrono/submission.csv artifacts/candidate_r45_k8time --order r33 --chrono 8 --final time
echo "artifacts/candidate_r54_maxep70"
$PY $D/build_pump_evidence_variant.py artifacts/candidate_r45_k8time/submission.csv artifacts/candidate_r54_maxep70 --order r33 --chrono 8 --final time --maxep 0.70
echo "artifacts/candidate_r58_77"
$PY $D/build_pump_evidence_variant.py artifacts/candidate_r54_maxep70/submission.csv artifacts/candidate_r58_77 --order r33 --chrono 8 --final time --maxep 0.7 --pairs 77
echo "pump_pu_scores.parquet"
$PY $D/pump_pu_classifier.py
echo "evidence_strength_pairs.parquet"
$PY $D/step60_evidence_strength_pairs.py
echo "artifacts/candidate_r70_evlift"
$PY $D/step70_r70_evlift.py
echo "artifacts/candidate_r72_evlift55, evidence_strength_r30.parquet"
$PY $D/step80_r72_evlift55.py
echo "artifacts/candidate_r75_other"
$PY $D/step90_r75_other.py
echo "artifacts/candidate_r77_keepfam"
$PY $D/step100_r77_keepfam.py
echo "artifacts/candidate_r80_keepfam_r33ev (+ restored_members.json)"
$PY $D/step110_r80_keepfam_r33ev.py
echo "artifacts/candidate_r88_zone"
$PY $D/step120_r88_zone.py
echo "artifacts/candidate_r92_prim4"
$PY $D/build_pump_evidence_variant.py artifacts/candidate_r88_zone/submission.csv artifacts/tmp_p4 --order r33 --chrono 8 --final time --maxep 0.7 --pairs 77 --pufallback 1 --nprim 4 && $PY $D/finalize_on_base.py artifacts/tmp_p4 artifacts/candidate_r92_prim4 artifacts/candidate_r88_zone/submission.csv && rm -rf artifacts/tmp_p4
echo "artifacts/candidate_r95_reverse"
$PY $D/build_pump_slots45.py artifacts/candidate_r92_prim4/submission.csv artifacts/candidate_r95_reverse --fb reverse
echo "artifacts/candidate_r103_final_inter"
$PY $D/build_pump_slots45.py artifacts/candidate_r95_reverse/submission.csv artifacts/candidate_r103_final_inter --fb reverse --nprim 3 --revfilter 1 --interleave 1
echo "artifacts/candidate_r104_revk5"
$PY $D/build_pump_slots45.py artifacts/candidate_r95_reverse/submission.csv artifacts/candidate_r104_revk5 --fb reverse --nprim 3 --revfilter 1 --interleave 1 --revk 5
