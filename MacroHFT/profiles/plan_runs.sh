#!/usr/bin/env bash
# [TradeMaster] Write the staged job lists for the profile comparison (does NOT run anything).
#
#   MacroHFT/profiles/plan_runs.sh [OUT_DIR] [SEEDS]      default: /tmp/macro_profiles, "12345 23456 34567"
#
# Stage 1 decompose (one job per dataset), stage 2 the six sub-agents per dataset, stage 3 the hyper-agent
# per dataset and seed. Each stage depends on the previous one; run them in order on the BOX, inside our
# allocation of 16 CPUs / 24 GB (box/runjobs.sh enforces it; the other 16 CPUs / 24 GB belong to SIMONS):
#   box/runjobs.sh -j 8  -m 2G    OUT_DIR/1_decompose.tsv      (16 GB)
#   box/runjobs.sh -j 16 -m 1500M OUT_DIR/2_train_low.tsv      (24 GB)
#   box/runjobs.sh -j 12 -m 2G    OUT_DIR/3_train_high.tsv     (24 GB; a full 1e6 replay buffer ~1.4 GB at 15m)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${1:-/tmp/macro_profiles}; SEEDS=${2:-"12345 23456 34567"}
mkdir -p "$OUT"; : > "$OUT/1_decompose.tsv"; : > "$OUT/2_train_low.tsv"; : > "$OUT/3_train_high.tsv"
for ds in $(cd "$HERE/data" && ls -d *_*_* 2>/dev/null | sort); do
  [[ -f $HERE/data/$ds/run.env ]] || continue
  EMIT_JOBS=1 DATASET=$ds DEVICE=cpu "$HERE/run.sh" decompose >> "$OUT/1_decompose.tsv"
  EMIT_JOBS=1 DATASET=$ds DEVICE=cpu "$HERE/run.sh" train-low >> "$OUT/2_train_low.tsv"
  for s in $SEEDS; do
    EMIT_JOBS=1 DATASET=$ds DEVICE=cpu "$HERE/run.sh" train-high -- --seed "$s" --exp "seed$s" \
      | sed "s|^\([^\t]*\)|\1_seed$s|" >> "$OUT/3_train_high.tsv"
  done
done
wc -l "$OUT"/*.tsv
