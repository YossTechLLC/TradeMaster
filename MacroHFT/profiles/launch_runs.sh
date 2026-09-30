#!/usr/bin/env bash
# [TradeMaster] Run the staged profile comparison written by plan_runs.sh, in order, inside our BOX allocation
# (16 CPUs / 24 GB, enforced by box/runjobs.sh). Stops at the first stage with a failed job.
#
#   MacroHFT/profiles/launch_runs.sh QUEUE_DIR            foreground
#   nohup MacroHFT/profiles/launch_runs.sh QUEUE_DIR > QUEUE_DIR/launch.log 2>&1 &    survives ssh disconnects
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
Q=${1:?usage: launch_runs.sh QUEUE_DIR}
stage() { # stage <file> <jobs> <mem>
  echo "=== [$(date '+%F %T')] $1 (-j $2 -m $3)"
  box/runjobs.sh -j "$2" -m "$3" "$Q/$1" || { echo "=== stage $1 had failures; stopping (see $Q/${1%.tsv}_logs/summary.tsv)"; exit 1; }
}
stage 1_decompose.tsv  8  2G
stage 2_train_low.tsv  16 1500M
stage 3_train_high.tsv 12 2G
echo "=== [$(date '+%F %T')] all stages done; results in MacroHFT/result/high_level/BTCUSDT_*_*/seed*/"
