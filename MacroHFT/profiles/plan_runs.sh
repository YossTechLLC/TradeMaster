#!/usr/bin/env bash
# [TradeMaster] Write the staged job lists for the profile comparison (does NOT run anything).
#
#   MacroHFT/profiles/plan_runs.sh [OUT_DIR] [SEEDS]      default: /tmp/macro_profiles, "12345 23456 34567"
#
# [TradeMaster] optional trading layer (see trading/README.md): TRADING_ARGS="--trade_mode long_short --venue kraken_us_perp
# --leverage 5 --turnover_penalty_bps 5" is appended to the train-low and train-high jobs, and TRADING_TAG (e.g. ls5) is
# suffixed to their job ids. Stage 2 and 3 must use the same TRADING_ARGS (results go to result/*/<dataset>@<mode-venue-L-tp>/).
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
read -r -a TRADING <<<"${TRADING_ARGS:-}"; TAG=${TRADING_TAG:+_$TRADING_TAG}  # [TradeMaster]
mkdir -p "$OUT"; : > "$OUT/1_decompose.tsv"; : > "$OUT/2_train_low.tsv"; : > "$OUT/3_train_high.tsv"
for ds in $(cd "$HERE/data" && ls -d *_*_* 2>/dev/null | sort); do
  [[ -f $HERE/data/$ds/run.env ]] || continue
  EMIT_JOBS=1 DATASET=$ds DEVICE=cpu "$HERE/run.sh" decompose >> "$OUT/1_decompose.tsv"
  EMIT_JOBS=1 DATASET=$ds DEVICE=cpu "$HERE/run.sh" train-low ${TRADING[@]:+-- "${TRADING[@]}"} \
    | sed "s|^\([^\t]*\)|\1$TAG|" >> "$OUT/2_train_low.tsv"  # [TradeMaster]
  for s in $SEEDS; do
    EMIT_JOBS=1 DATASET=$ds DEVICE=cpu "$HERE/run.sh" train-high -- --seed "$s" --exp "seed$s" "${TRADING[@]}" \
      | sed "s|^\([^\t]*\)|\1_seed$s$TAG|" >> "$OUT/3_train_high.tsv"
  done
done
wc -l "$OUT"/*.tsv
