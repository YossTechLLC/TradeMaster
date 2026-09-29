#!/usr/bin/env bash
# TradeMaster launcher for EarnHFT (AAAI'24).
#
# Wraps the upstream pipeline (data_preprocess/*.sh, EarnHFT_Algorithm/script/<PAIR>/*.sh)
# for a single-GPU machine: runs each step from the directory upstream expects,
# uses the shared .venv-hft interpreter, runs in the foreground, and tees logs.
# Upstream scripts are left untouched for reference.
#
# Usage: ./run.sh <stage> [args] [-- extra python args]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HERE/../.venv-hft/bin/python}"
ALGO="$HERE/EarnHFT_Algorithm"
PREP="$HERE/data_preprocess"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BETAS="${BETAS:--10 -90 30 100}"   # low-level sampling preferences used upstream
SEED=12345
MAX_VALID_EPOCHS="${MAX_VALID_EPOCHS:-50}"  # upstream valid_multi.sh caps at 50 epochs per beta
MAX_TEST_EPOCHS="${MAX_TEST_EPOCHS:-80}"    # upstream high_level/test.sh caps at 80

usage() {
  cat <<EOF
EarnHFT pipeline (run stages in order):

 Data (in data_preprocess/, needs a Tardis API key for download):
  download   <PAIR> <START> <END>   Tardis book_snapshot_5 + trades  (needs: pip install tardis-dev nest_asyncio)
  preprocess <PAIR> <START> <END>   concat_clean -> create_feature -> merge_new -> calculate_ic,
                                     then installs df.feather + feature lists into EarnHFT_Algorithm/data/
 Algorithm (in EarnHFT_Algorithm/):
  split      <PAIR>                 train/valid/test split + training chunks
  label      <PAIR>                 market-dynamics labels for the valid set (strategy-pool selection)
  train-low  <PAIR> [BETA]          stage I/II: second-level DDQN w/ Q-teacher (all $BETAS if BETA omitted)
  valid-low  <PAIR>                 evaluate every low-level epoch on labelled valid data (5 initial positions)
  pick       <PAIR>                 stage II: build the agent pool -> result_risk/<PAIR>/potential_model
  train-high <PAIR>                 stage III: minute-level router (DQN over the pool)
  test-high  <PAIR> [EPOCH_DIR]     test router epochs (all under result_risk/<PAIR>/high_level/seed_12345 if omitted)

 PAIR: BTCUSDT | ETHUSDT | GALAUSDT | BTCTUSD (router agent pools are hard-coded for these four
 in RL/agent/high_level/dqn_position.py). Anything after '--' is forwarded to the python script.
 Env overrides: PY, CUDA_VISIBLE_DEVICES, BETAS, MAX_HOLD, TC, MAX_VALID_EPOCHS (50), MAX_TEST_EPOCHS (80).
EOF
}

# Per-pair trading params mirroring upstream script/<PAIR>/*.sh
# (the max position should be worth ~300 USD; BTCTUSD is a zero-fee pair).
pair_args() {
  local hold tc
  case "$1" in
    ETHUSDT)  hold=0.1 ;  tc=0.00015 ;;
    GALAUSDT) hold=4000 ; tc=0.00015 ;;
    BTCTUSD)  hold=0.01 ; tc=0 ;;
    *)        hold=0.01 ; tc=0.00015 ;;
  esac
  echo "--max_holding_number ${MAX_HOLD:-$hold} --transcation_cost ${TC:-$tc}"
}

# run <workdir> <logfile relative to workdir> <script> [args...]
run() {
  local wd=$1 log=$2; shift 2
  mkdir -p "$wd/$(dirname "$log")"
  echo ">>> [$(date '+%F %T')] (cd ${wd#"$HERE"/}) python $*" | tee -a "$wd/$log"
  (cd "$wd" && "$PY" -u "$@" 2>&1 | tee -a "$log")
}

stage=${1:-help}; shift || true
# split positional args from pass-through args after `--`
POS=(); EXTRA=()
while [[ $# -gt 0 ]]; do
  if [[ $1 == "--" ]]; then shift; EXTRA=("$@"); break; fi
  POS+=("$1"); shift
done
need() { [[ ${#POS[@]} -ge $1 ]] || { echo "stage '$stage' needs: $2" >&2; exit 2; }; }

case "$stage" in
  download)
    need 3 "<PAIR> <START> <END>"
    for t in book_snapshot_5 trades; do
      run "$PREP" "download_code/log/${POS[0]}_$t.log" download_code/download.py \
        --symbols "${POS[0]}" --data_types "$t" --start_date "${POS[1]}" --end_date "${POS[2]}" "${EXTRA[@]}"
    done ;;

  preprocess)
    need 3 "<PAIR> <START> <END>"
    P=${POS[0]}; R="${POS[1]}-${POS[2]}"
    run "$PREP" "preprocess/log/${P}_clean.log"  preprocess/concat_clean.py \
      --symbols "$P" --start_date "${POS[1]}" --end_date "${POS[2]}"
    run "$PREP" "preprocess/log/${P}_create.log" preprocess/create_feature.py \
      --data_path "preprocess/concat_clean/$P/$R"
    run "$PREP" "preprocess/log/${P}_merge.log"  preprocess/merge_new.py \
      --data_path "preprocess/create_features/$P/$R/beatrate_0.0001"
    run "$PREP" "ic_analysis/log/${P}_ic.log"    ic_analysis/calculate_ic.py \
      --data_path "preprocess/merge/$P/$R/df.feather"
    # Hand-off to the algorithm side (upstream leaves this as a manual copy).
    mkdir -p "$ALGO/data/$P" "$ALGO/data/feature"
    cp -v "$PREP/preprocess/merge/$P/$R/df.feather" "$ALGO/data/$P/df.feather"
    cp -v "$PREP/ic_analysis/feature_analysis/$P/${POS[1]}_${POS[2]}/second_feature.npy" \
          "$PREP/ic_analysis/feature_analysis/$P/${POS[1]}_${POS[2]}/minitue_feature.npy" "$ALGO/data/feature/"
    echo "NOTE: data/feature/*.npy is shared by all pairs (upstream design) - the last preprocessed pair wins." ;;

  split)
    need 1 "<PAIR>"
    run "$ALGO" "log/data/split/${POS[0]}.log" data/split_data.py --data_path "data/${POS[0]}" "${EXTRA[@]}" ;;

  label)
    need 1 "<PAIR>"
    run "$ALGO" "log/data/split_valid/${POS[0]}.log" tool/slice_model.py \
      --data_path "data/${POS[0]}/valid.feather" "${EXTRA[@]}" ;;

  train-low)
    need 1 "<PAIR> [BETA]"
    P=${POS[0]}
    for b in ${POS[1]:-$BETAS}; do
      # shellcheck disable=SC2046
      run "$ALGO" "log/train/$P/low_level/beta_$b.log" RL/agent/low_level/ddqn_pes_risk_aware.py \
        --beta "$b" --train_data_path "data/$P/train" --dataset_name "$P" $(pair_args "$P") "${EXTRA[@]}"
    done ;;

  valid-low)
    need 1 "<PAIR>"
    P=${POS[0]}
    for b in $BETAS; do
      root="$ALGO/result_risk/$P/beta_$(printf '%.1f' "$b")_risk_bond_0.1/seed_$SEED"
      [[ -d $root ]] || { echo "skip: $root not found (train-low first)"; continue; }
      for ep in $(ls "$root" | grep '^epoch_' | sort -t_ -k2 -n | head -n "$MAX_VALID_EPOCHS"); do
        for a in 0 1 2 3 4; do
          # shellcheck disable=SC2046
          run "$ALGO" "log/pick/$P/beta_$b/position_${a}_$ep.log" RL/agent/low_level/test_ddqn.py \
            --test_path "${root#"$ALGO"/}/$ep" --initial_action "$a" --test_df_path "data/$P/valid" \
            $(pair_args "$P") "${EXTRA[@]}"
        done
      done
    done ;;

  pick)
    need 1 "<PAIR>"
    run "$ALGO" "log/pick/${POS[0]}/high_level.log" analysis/pick_agent/pick_agent_position.py \
      --root_path "result_risk/${POS[0]}" --save_path "result_risk/${POS[0]}/potential_model" "${EXTRA[@]}" ;;

  train-high)
    need 1 "<PAIR>"
    # shellcheck disable=SC2046
    run "$ALGO" "log/train/${POS[0]}/high_level/log.log" RL/agent/high_level/dqn_position.py \
      --train_data_path "data/${POS[0]}/train.feather" --dataset_name "${POS[0]}" $(pair_args "${POS[0]}") "${EXTRA[@]}" ;;

  test-high)
    need 1 "<PAIR> [EPOCH_DIR]"
    P=${POS[0]}
    if [[ -n ${POS[1]:-} ]]; then eps=("${POS[1]}")
    else
      root="result_risk/$P/high_level/seed_$SEED"
      mapfile -t eps < <(ls "$ALGO/$root" | grep '^epoch_' | sort -t_ -k2 -n | head -n "$MAX_TEST_EPOCHS" | sed "s|^|$root/|")
    fi
    for ep in "${eps[@]}"; do
      # shellcheck disable=SC2046
      run "$ALGO" "log/test/$P/test_$(basename "$ep").log" RL/agent/high_level/test_dqn_position.py \
        --test_path "$ep" --dataset_name "$P" --valid_data_path "data/$P/valid.feather" \
        --test_data_path "data/$P/test.feather" $(pair_args "$P") "${EXTRA[@]}"
    done ;;

  help|-h|--help) usage ;;
  *) echo "unknown stage: $stage" >&2; usage; exit 2 ;;
esac
