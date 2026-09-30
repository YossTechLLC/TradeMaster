#!/usr/bin/env bash
# Throughput benchmark for docs/bottlenecks.md, run on the BOX from the repo root (CPU, 1 thread per process).
# Needs the fixtures: box/bench/build_data.py macro 1m 2024-01-01 30 12 9 BTCUSDT; ... earn 2024-03-01 4;
# box/bench/golden.py prepare. Every job goes through box/runjobs.sh (per-job MemoryMax, 48 GB budget).
#
#   box/bench/box_bench.sh [macro|earn|scale|all]
set -uo pipefail
cd "$(dirname "$0")/../.."
TM=$PWD; PY=$TM/.venv-hft/bin/python; OUT=${BENCH_OUT:-/tmp/tm_bench}; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES= DEVICE=cpu
what=${1:-all}
ts() { $PY -u box/bench/ts.py; }

if [[ $what == macro || $what == all ]]; then
  echo "== MacroHFT: decompose, six sub-agents in parallel (1 epoch), hyper-agent (1 epoch) on BTCUSDT"
  rm -rf MacroHFT/result/low_level/BTCUSDT MacroHFT/result/high_level/BTCUSDT/bench
  ( time DATASET=BTCUSDT MacroHFT/run.sh decompose > "$OUT/macro_decompose.log" 2>&1 ) 2>&1 | grep real
  EMIT_JOBS=1 DATASET=BTCUSDT MacroHFT/run.sh train-low -- --epoch_number 1 > "$OUT/macro_low.tsv"
  sed -i "s|\$| 2>\&1 \| $PY -u $TM/box/bench/ts.py|" "$OUT/macro_low.tsv"
  box/runjobs.sh -j 6 -m 1500M "$OUT/macro_low.tsv" | tail -1
  printf 'high\tcd %s/MacroHFT && %s -u RL/agent/high_level.py --dataset BTCUSDT --device cpu --epoch_number 1 --exp bench 2>&1 | %s -u %s/box/bench/ts.py\n' \
    "$TM" "$PY" "$PY" "$TM" > "$OUT/macro_high.tsv"
  box/runjobs.sh -j 1 -m 3G "$OUT/macro_high.tsv" | tail -1
fi

if [[ $what == earn || $what == all ]]; then
  echo "== EarnHFT: split, label, train-low (1 beta, 8 samples), valid-low, pick, router (2 passes), test-high on BTCUSDT"
  A=EarnHFT/EarnHFT_Algorithm
  rm -rf $A/result_risk/BTCUSDT
  EarnHFT/run.sh split BTCUSDT > "$OUT/earn_split.log" 2>&1
  printf 'label\tcd %s && EarnHFT/run.sh label BTCUSDT 2>&1 | %s -u box/bench/ts.py | grep -v "it/s"\n' "$TM" "$PY" > "$OUT/earn_label.tsv"
  box/runjobs.sh -j 1 -m 4G "$OUT/earn_label.tsv" | tail -1
  EMIT_JOBS=1 BETAS="30" EarnHFT/run.sh train-low BTCUSDT -- --num_sample 8 > "$OUT/earn_low.tsv"
  sed -i "s|\$| 2>\&1 \| $PY -u $TM/box/bench/ts.py|" "$OUT/earn_low.tsv"
  box/runjobs.sh -j 1 -m 3500M "$OUT/earn_low.tsv" | tail -1
  EMIT_JOBS=1 BETAS="30" EarnHFT/run.sh valid-low BTCUSDT > "$OUT/earn_valid.tsv"
  box/runjobs.sh -j 10 -m 1500M "$OUT/earn_valid.tsv" | tail -1
  EarnHFT/run.sh pick BTCUSDT > "$OUT/earn_pick.log" 2>&1
  EMIT_JOBS=1 EarnHFT/run.sh train-high BTCUSDT -- --num_sample 2 > "$OUT/earn_router.tsv"
  sed -i "s|\$| 2>\&1 \| $PY -u $TM/box/bench/ts.py|" "$OUT/earn_router.tsv"
  box/runjobs.sh -j 1 -m 3G "$OUT/earn_router.tsv" | tail -1
  EMIT_JOBS=1 EarnHFT/run.sh test-high BTCUSDT > "$OUT/earn_test.tsv"
  box/runjobs.sh -j 2 -m 2G "$OUT/earn_test.tsv" | tail -1
fi

if [[ $what == scale || $what == all ]]; then
  echo "== Scaling: N copies of a 1-epoch hyper-agent job on BTCGOLD (6000 train + 1500 val + 1500 test bars)"
  for n in 1 8 16 32; do
    : > "$OUT/scale_$n.tsv"
    for i in $(seq 1 "$n"); do
      printf 'h%s\tcd %s/MacroHFT && %s -u RL/agent/high_level.py --dataset BTCGOLD --device cpu --epoch_number 1 --exp scale%s --seed %s --subagent_path ./result/low_level/ETHUSDT/best_model\n' \
        "$i" "$TM" "$PY" "$n" "$i" >> "$OUT/scale_$n.tsv"
    done
    start=$(date +%s.%N)
    box/runjobs.sh -j "$n" -m 1400M "$OUT/scale_$n.tsv" > /dev/null
    end=$(date +%s.%N)
    $PY -c "import sys; n,t=int(sys.argv[1]),float(sys.argv[3])-float(sys.argv[2]); print(f'{n:>3} processes: {t:6.1f} s wall, {n*9000/t:8.0f} bars/s aggregate')" "$n" "$start" "$end"
  done
  rm -rf MacroHFT/result/high_level/BTCGOLD/scale*
fi
