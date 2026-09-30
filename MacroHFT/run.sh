#!/usr/bin/env bash
# TradeMaster launcher for MacroHFT (KDD'24).
#
# Wraps upstream scripts/*.sh for a single-GPU machine: runs from the MacroHFT root
# (all paths are ./data, ./result relative), uses the shared .venv-hft interpreter,
# pins every job to one device, runs in the foreground, and tees logs to logs/.
# Upstream scripts/ are left untouched for reference.
#
# Usage: ./run.sh <stage> [args] [-- extra python args]
#        EMIT_JOBS=1 ./run.sh <stage> ... > jobs.tsv   # list the stage's jobs instead of running them,
#        ../box/runjobs.sh -j N -m MEM jobs.tsv       # then run them in parallel (BOX, CPU)
set -euo pipefail
export TM_THREADS="${TM_THREADS:-1}"   # threads per python process (the scripts pin torch/OMP/MKL to it)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HERE/../.venv-hft/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
DATASET="${DATASET:-ETHUSDT}"   # data/<DATASET>/, result/*/<DATASET>/

# The six sub-agents from upstream scripts/low_level.sh: "<clf> <label> <alpha>"
SUBAGENTS=(
  "slope label_1 1"   # bear
  "slope label_2 4"   # flat trend
  "slope label_3 0"   # bull
  "vol   label_1 4"   # stable
  "vol   label_2 1"   # medium volatility
  "vol   label_3 1"   # volatile
)

usage() {
  cat <<EOF
MacroHFT pipeline (run from anywhere; stages in order):

  build PROFILE TIMEFRAME        Step 0 (input profiles): download Binance spot klines, compute the profile's
                                  features, scale on train -> data/<SYMBOL>_<TIMEFRAME>_<PROFILE>/.
                                  'build --all' builds every profiles/*.yaml x timeframe. Use that name as DATASET.
  admit                          Admission test of \$DATASET's features on train (profiles/admit.py)
  decompose                      Step 1: chunk data/\$DATASET/df_{train,val,test}.feather and label chunks
                                  by trend (slope) and volatility -> data/\$DATASET/{train,val,test,whole}/
  train-low [CLF LABEL ALPHA]    Step 2: phase I sub-agents (DDQN + conditional adapter). With no args,
                                  trains all six sequentially; e.g. 'train-low slope label_1 1' for one.
                                  Writes result/low_level/\$DATASET/best_model/<clf>/<n>/best_model.pkl
                                  (for ETHUSDT this overwrites the pretrained checkpoints shipped with the repo).
  train-high                     Step 3: phase II hyper-agent with memory, mixing the six sub-agents
                                  -> result/high_level/\$DATASET/<exp>/seed_<n>/
                                  Sub-agents come from result/low_level/\$DATASET/best_model/;
                                  override with '-- --subagent_path DIR'.

  Pretrained sub-agents are already in result/low_level/ETHUSDT/best_model/, so after
  'decompose' you can go straight to 'train-high'.
  Data: download the ETHUSDT folder from the Google Drive link in readme.md into data/.
  Env overrides: PY, DEVICE (default cuda:0), DATASET (default ETHUSDT).
  Anything after '--' is forwarded to python.
EOF
}

run() { # run <log> <script> [args...]
  local log=$1; shift
  if [[ -n ${EMIT_JOBS:-} ]]; then  # print "<job id>\t<command>" for box/runjobs.sh instead of running
    printf '%s\t' "$(echo "${log%.log}" | tr '/' '_')"; printf 'cd %q && ' "$HERE"
    # the dataset's run.env settings must reach decompose when the job runs later
    [[ -n ${MACRO_CHUNK_SIZE:-} ]] && printf 'MACRO_CHUNK_SIZE=%q MACRO_CONTEXT_WINDOW=%q ' "$MACRO_CHUNK_SIZE" "$MACRO_CONTEXT_WINDOW"
    printf '%q ' "$PY" -u "$@"; echo
    return
  fi
  mkdir -p "$HERE/$(dirname "$log")"
  echo ">>> [$(date '+%F %T')] python $*" | tee -a "$HERE/$log"
  (cd "$HERE" && "$PY" -u "$@" 2>&1 | tee -a "$log")
}

stage=${1:-help}; shift || true
POS=(); EXTRA=()
while [[ $# -gt 0 ]]; do
  if [[ $1 == "--" ]]; then shift; EXTRA=("$@"); break; fi
  POS+=("$1"); shift
done

# Per-dataset run settings written by profiles/build.py (chunk size, context window, memory size)
if [[ -f $HERE/data/$DATASET/run.env ]]; then
  set -a; source "$HERE/data/$DATASET/run.env"; set +a
fi
read -r -a HIGH_ARGS <<<"${MACRO_HIGH_ARGS:-}"

case "$stage" in
  build)
    if [[ ${POS[0]:-} == --all ]]; then run "logs/build.log" profiles/build.py --all "${EXTRA[@]}"
    else
      [[ ${#POS[@]} -eq 2 ]] || { echo "build takes PROFILE TIMEFRAME, or --all" >&2; exit 2; }
      run "logs/build_${POS[0]}_${POS[1]}.log" profiles/build.py --profile "${POS[0]}" --timeframe "${POS[1]}" "${EXTRA[@]}"
    fi ;;

  admit)
    run "logs/admit_$DATASET.log" profiles/admit.py --dataset "$DATASET" "${EXTRA[@]}" ;;

  decompose)
    for f in df_train df_val df_test; do
      [[ -f $HERE/data/$DATASET/$f.feather ]] || { echo "missing data/$DATASET/$f.feather - see readme.md (Google Drive)" >&2; exit 1; }
    done
    run "logs/decomposition_$DATASET.log" preprocess/decomposition.py "$DATASET" ;;

  train-low)
    case ${#POS[@]} in
      0) jobs=("${SUBAGENTS[@]}") ;;
      3) jobs=("${POS[*]}") ;;
      *) echo "train-low takes no arguments (all six) or exactly: CLF LABEL ALPHA" >&2; exit 2 ;;
    esac
    for j in "${jobs[@]}"; do
      read -r clf label alpha <<<"$j"
      run "logs/low_level/$DATASET/${clf}_${label#label_}.log" RL/agent/low_level.py \
        --alpha "$alpha" --clf "$clf" --dataset "$DATASET" --device "$DEVICE" --label "$label" "${EXTRA[@]}"
    done ;;

  train-high)
    for f in train val test; do  # (not checked when only listing jobs: decompose runs in an earlier stage)
      [[ -n ${EMIT_JOBS:-} || -f $HERE/data/$DATASET/whole/$f.feather ]] || { echo "missing data/$DATASET/whole/$f.feather - run './run.sh decompose' first" >&2; exit 1; }
    done
    run "logs/high_level/$DATASET.log" RL/agent/high_level.py \
      --dataset "$DATASET" --device "$DEVICE" "${HIGH_ARGS[@]}" "${EXTRA[@]}" ;;

  help|-h|--help) usage ;;
  *) echo "unknown stage: $stage" >&2; usage; exit 2 ;;
esac
