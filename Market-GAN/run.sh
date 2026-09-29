#!/usr/bin/env bash
# TradeMaster launcher for Market-GAN (AAAI'24).
#
# Runs the upstream shell scripts unchanged in content (they hold the paper's
# hyper-parameters) but: from the directory each one expects, with the shared
# .venv-hft interpreter first on PATH, every hard-coded cuda:N rewritten to one
# device, in the foreground, with logs teed to logs/.
#
# Usage: ./run.sh <stage>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$(cd "$HERE/.." && pwd)/.venv-hft}"
DEVICE="${DEVICE:-cuda:0}"
export PATH="$VENV/bin:$PATH"

usage() {
  cat <<EOF
Market-GAN pipeline (stages in order; see README.md for the paper mapping):

 I) Market Dynamics Modeling
  mdm              Label data/DJI/DJI_data.csv into bear/flat/bull dynamics
                    -> data/DJI/DJI_50/tic/DJI_data_labeled_slice_and_merge_model_3dynamics_minlength50_quantile_labeling.csv
                    (rewrites MarketDynamicsModeling/configs/market_dynamics_modeling/djia.py with result paths)
 II) Market-GAN
  pretrain         Pre-train the condition supervisors (Pretrain_DJI_V2_50.sh)      -> output/DJI_pretrain_50
  train            Two-stage training: AE/supervisor + adversarial (DJI_V2_RT_train.sh) -> output/DJ30_V2_RT
  evaluate         Evaluate the trained model (Evaluate_DJI_V2_RT.sh)
  plot             Visualise generated vs real data (Plot_DJI_V2_RT.sh)
 III) Downstream use
  export           Export model info for inference (DJI_V2_RT_info_export.sh)
  generate         Generate augmented training data (service/run_MarketGAN.sh)
                    -> downstream_tasks/data/downstream_tasks/data/MarketGAN
  prep-real        Prepare the real-data baseline for forecasting (downstream_tasks/data/DJI.sh)
  forecast-real    Forecasting models on real data only        (downstream_tasks/run_real*.sh)
  forecast-gan     Forecasting models on real + Market-GAN data (downstream_tasks/run_MarketGAN*.sh)

 Skip 'pretrain'/'train' by downloading the checkpoint (SharePoint link in README.md) into output/.
 Env overrides: VENV, DEVICE (default cuda:0).
EOF
}

# sh_run <workdir relative to HERE> <script relative to workdir>
sh_run() {
  local wd="$HERE/$1" script=$2 log
  log="$HERE/logs/$(basename "${script%.sh}").log"
  mkdir -p "$HERE/logs"
  echo ">>> [$(date '+%F %T')] (cd $1) $script  [device=$DEVICE, python=$(command -v python)]" | tee -a "$log"
  (cd "$wd" && sed -E "s/cuda:[0-9]+/$DEVICE/g" "$script" | bash 2>&1 | tee -a "$log")
}

stage=${1:-help}

case "$stage" in
  mdm)
    mkdir -p "$HERE/logs"
    (cd "$HERE/MarketDynamicsModeling" && python -u tools/market_dynamics_labeling/run.py 2>&1 | tee -a "$HERE/logs/mdm.log") ;;
  pretrain)      sh_run . Pretrain_DJI_V2_50.sh ;;
  train)         sh_run . DJI_V2_RT_train.sh ;;
  evaluate)      sh_run . Evaluate_DJI_V2_RT.sh ;;
  plot)          sh_run . Plot_DJI_V2_RT.sh ;;
  export)        sh_run . DJI_V2_RT_info_export.sh ;;
  generate)      sh_run . service/run_MarketGAN.sh ;;
  prep-real)     sh_run downstream_tasks/data DJI.sh ;;
  forecast-real) for s in run_real.sh run_real_GRU.sh run_real_LSTM.sh run_real_RNN.sh run_real_TCN.sh; do sh_run downstream_tasks "$s"; done ;;
  forecast-gan)  for s in run_MarketGAN.sh run_MarketGAN_GRU.sh run_MarketGAN_LSTM.sh run_MarketGAN_RNN.sh run_MarketGAN_TCN.sh; do sh_run downstream_tasks "$s"; done ;;
  help|-h|--help) usage ;;
  *) echo "unknown stage: $stage" >&2; usage; exit 2 ;;
esac
