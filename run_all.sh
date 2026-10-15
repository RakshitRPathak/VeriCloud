#!/usr/bin/env bash
# Reproduce every number, table and figure in results/ (about 1-2 h on 2 CPU cores).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/figures results/tables results/sample_reports
python3 -m pytest -q tests 2>&1 | tee results/tests.log
cd experiments
for e in fig_architecture exp1_main exp2_adversarial exp2b_adaptive exp3_source_learning exp4_ablation exp5_calibration_reports exp6_efficiency; do
  echo "=== $e ==="; python3 -u $e.py
done 2>&1 | tee ../results/run_all.log
