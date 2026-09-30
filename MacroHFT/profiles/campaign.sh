#!/usr/bin/env bash
# [TradeMaster] End-to-end long/short campaign on the BOX, inside our 16 CPU / 24 GB allocation (box/runjobs.sh):
#   1 decompose -> 2 sub-agents -> 3 hyper-agents (seeds x penalties) -> 4 full signal export (val+test) -> 5 tm_risk replay.
# Each stage runs only if the previous one fully succeeded; progress goes to $Q/campaign.log ("=== ..." stage lines,
# runjobs per-job lines, and a final "=== CAMPAIGN DONE" or "=== CAMPAIGN STOPPED" line). Re-running resumes: a stage
# whose summary.tsv shows every job succeeded is skipped.
#
#   QUEUE=box/queue/ls5m_L3 DATASETS="BTCUSDT_5m_xt10 BTCUSDT_5m_upstream_real" LEVERAGE=3 PENALTIES="0 5 10" \
#   SEEDS="12345 23456 34567" ARMS=identity_L3,constant_1x_L3,vol_target_25_L3,default_35dd_L3 \
#   setsid nohup MacroHFT/profiles/campaign.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TM=$PWD
Q=${QUEUE:?QUEUE dir}; DATASETS=${DATASETS:?}; LEVERAGE=${LEVERAGE:?}; PENALTIES=${PENALTIES:-"0 5 10"}
SEEDS=${SEEDS:-"12345 23456 34567"}; VENUE=${VENUE:-kraken_us_perp}; ARMS=${ARMS:?}
J_DEC=${J_DEC:-8}; M_DEC=${M_DEC:-2G}; J_LOW=${J_LOW:-16}; M_LOW=${M_LOW:-1500M}
J_HIGH=${J_HIGH:-12}; M_HIGH=${M_HIGH:-2G}; J_EXP=${J_EXP:-12}; M_EXP=${M_EXP:-2G}; REPLAY_PROCS=${REPLAY_PROCS:-12}
mkdir -p "$Q"; Q=$(cd "$Q" && pwd); LOG="$Q/campaign.log"   # absolute from here on
say() { echo "=== [$(date -u '+%F %T')] $*" >> "$LOG"; }

plan() {
  : > "$Q/1_decompose.tsv"; : > "$Q/2_train_low.tsv"; : > "$Q/3_train_high.tsv"; : > "$Q/4_export.tsv"; : > "$Q/5_replay.tsv"
  local first=1
  for tp in $PENALTIES; do
    local args="--trade_mode long_short --venue $VENUE --leverage $LEVERAGE --turnover_penalty_bps $tp"
    DATASETS="$DATASETS" TRADING_ARGS="$args" TRADING_TAG="tp$tp" MacroHFT/profiles/plan_runs.sh "$Q/plan_tp$tp" "$SEEDS" > /dev/null
    (( first )) && cp "$Q/plan_tp$tp/1_decompose.tsv" "$Q/1_decompose.tsv"; first=0
    cat "$Q/plan_tp$tp/2_train_low.tsv" >> "$Q/2_train_low.tsv"
    cat "$Q/plan_tp$tp/3_train_high.tsv" >> "$Q/3_train_high.tsv"
    local tag="long_short-$VENUE-L$(printf '%g' "$LEVERAGE")-tp$tp"
    for ds in $DATASETS; do for s in $SEEDS; do
      printf 'export_%s_%s_seed%s\tcd %q/MacroHFT && %q -u trading/export_signal.py --run_dir %q --dataset %s --splits val test\n' \
        "$ds" "tp$tp" "$s" "$TM" "$TM/.venv-hft/bin/python" "result/high_level/$ds@$tag/seed$s/seed_$s" "$ds" >> "$Q/4_export.tsv"
    done; done
  done
  local globs=""; for ds in $DATASETS; do globs="$globs $ds"; done
  printf 'replay\tcd %q && for ds in%s; do %q box/bench/risk_replay_1h.py --runs_glob "MacroHFT/result/high_level/${ds}@long_short-%s-L%s-tp*/*/seed_*" --arms %s --procs %s --json %q/replay_${ds}.json > %q/replay_${ds}.md || exit 1; done\n' \
    "$TM" "$globs" "$TM/.venv-hft/bin/python" "$VENUE" "$(printf '%g' "$LEVERAGE")" "$ARMS" "$REPLAY_PROCS" "$Q" "$Q" >> "$Q/5_replay.tsv"
}

stage() { # stage <file> <jobs> <mem> [threads]
  local f=$1 j=$2 m=$3 t=${4:-1} n s
  n=$(grep -c $'\t' "$Q/$f"); s="$Q/${f%.tsv}_logs/summary.tsv"
  if [[ -f $s ]] && (( $(wc -l < "$s") == n )) && ! awk -F'\t' '$2 != 0 {bad=1} END {exit !bad}' "$s"; then
    say "$f: already complete ($n jobs), skipped"; return 0
  fi
  say "$f: $n jobs (-j $j -m $m -t $t)"
  box/runjobs.sh -j "$j" -m "$m" -t "$t" "$Q/$f" >> "$LOG" 2>&1 || { say "CAMPAIGN STOPPED at $f (see ${s})"; exit 1; }
}

say "CAMPAIGN START datasets=[$DATASETS] leverage=$LEVERAGE penalties=[$PENALTIES] seeds=[$SEEDS] venue=$VENUE arms=$ARMS"
[[ -f $Q/2_train_low.tsv ]] || plan
if [[ -n ${DRY_RUN:-} ]]; then wc -l "$Q"/*.tsv; say "DRY RUN: planned only"; exit 0; fi
stage 1_decompose.tsv "$J_DEC" "$M_DEC"
stage 2_train_low.tsv "$J_LOW" "$M_LOW"
stage 3_train_high.tsv "$J_HIGH" "$M_HIGH"
stage 4_export.tsv "$J_EXP" "$M_EXP"
stage 5_replay.tsv 1 8G "$REPLAY_PROCS"
say "CAMPAIGN DONE (replay tables: $Q/replay_*.md)"
