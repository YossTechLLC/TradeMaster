#!/usr/bin/env bash
# Run independent CPU jobs in parallel under a per-job memory cap (TradeMaster on the BOX).
#
#   box/runjobs.sh [-j JOBS] [-m MEM_PER_JOB] [-l LOG_DIR] [-t THREADS] [--no-scope] JOBFILE
#   box/runjobs.sh --check [-j JOBS] [-m MEM_PER_JOB] [-t THREADS]      preflight only, runs nothing
#
# JOBFILE: one job per line, "<job id>\t<shell command>" (what `EMIT_JOBS=1 <project>/run.sh <stage>` prints).
# Every job runs with TM_THREADS/OMP/MKL = THREADS (default 1) inside its own
#   systemd-run --user --scope -p MemoryMax=MEM -p MemorySwapMax=0
# so one runaway job is killed alone. A failed job does not stop the others; each job's output goes to
# LOG_DIR/<job id>.log, results to LOG_DIR/summary.tsv (id, exit code, seconds), and the script exits 1
# if any job failed.
#
# The BOX is shared with another agent (SIMONS compute), each side allocated 16 CPUs / 24 GB. Enforced here:
#   - every job runs inside one systemd slice (BOX_SLICE, default tmhft.slice) capped at BOX_MEM_BUDGET (24G)
#     and BOX_CPU_BUDGET CPUs (16) as a whole: the kernel holds the allocation even across several runjobs
#     invocations at once, and each job additionally has its own MemoryMax;
#   - the plan must fit: JOBS x MEM_PER_JOB <= BOX_MEM_BUDGET and JOBS x THREADS <= BOX_CPU_BUDGET;
#   - the host must currently have the planned memory available (MemAvailable), and >= 20 GiB disk free.
set -uo pipefail

JOBS=8; MEM=1500M; LOG_DIR=""; THREADS=1; SCOPE=1; CHECK=0
BUDGET="${BOX_MEM_BUDGET:-24G}"; CPU_BUDGET="${BOX_CPU_BUDGET:-16}"; SLICE="${BOX_SLICE:-tmhft.slice}"
MIN_FREE_GIB="${BOX_MIN_FREE_GIB:-20}"
while [[ $# -gt 0 ]]; do
  case $1 in
    -j) JOBS=$2; shift 2 ;;
    -m) MEM=$2; shift 2 ;;
    -l) LOG_DIR=$2; shift 2 ;;
    -t) THREADS=$2; shift 2 ;;
    --no-scope) SCOPE=0; shift ;;
    --check) CHECK=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
if (( CHECK )); then JOBFILE=""; LOG_DIR=${LOG_DIR:-.}
else
  JOBFILE=${1:?usage: runjobs.sh [-j N] [-m MEM] [-l LOG_DIR] [-t THREADS] [--no-scope] JOBFILE}
  [[ -f $JOBFILE ]] || { echo "no such job file: $JOBFILE" >&2; exit 2; }
  LOG_DIR=${LOG_DIR:-$(dirname "$JOBFILE")/$(basename "$JOBFILE" .tsv)_logs}
fi
mkdir -p "$LOG_DIR"

bytes() { numfmt --from=iec "${1%B}"; }
plan=$(( JOBS * $(bytes "$MEM") ))
if (( plan > $(bytes "$BUDGET") )); then
  echo "refusing: $JOBS jobs x $MEM > memory budget $BUDGET (lower -j or -m)" >&2; exit 2
fi
if (( JOBS * THREADS > CPU_BUDGET )); then
  echo "refusing: $JOBS jobs x $THREADS thread(s) > CPU budget $CPU_BUDGET" >&2; exit 2
fi
free_gib=$(( $(df --output=avail -k "$LOG_DIR" | tail -1) / 1048576 ))
if (( free_gib < MIN_FREE_GIB )); then
  echo "refusing: only ${free_gib} GiB free on $(df --output=target "$LOG_DIR" | tail -1) (need >= $MIN_FREE_GIB)" >&2; exit 2
fi
avail=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) * 1024 ))
if (( avail < plan )); then
  echo "refusing: the host has $(numfmt --to=iec $avail) available, the plan needs up to $(numfmt --to=iec $plan)" \
       "(the other agent / owner is using more than expected; wait or lower -j/-m)" >&2; exit 2
fi
if (( SCOPE )); then
  # the shared allocation: one slice for all our jobs, capped as a whole
  if ! systemctl --user set-property --runtime "$SLICE" MemoryMax="$BUDGET" MemorySwapMax=0 CPUQuota="$(( CPU_BUDGET * 100 ))%" 2>/dev/null \
     || ! systemd-run --user --scope --quiet --slice="$SLICE" -p MemoryMax=64M true 2>/dev/null; then
    echo "systemd user scopes/slices are unavailable here; rerun with --no-scope (no memory/CPU caps)" >&2; exit 2
  fi
fi
load=$(cut -d' ' -f1 /proc/loadavg)
echo "preflight ok: plan $JOBS x $MEM x $THREADS thread(s) in $SLICE (cap $BUDGET, $CPU_BUDGET CPUs);" \
     "host: $(numfmt --to=iec $avail) available, ${free_gib} GiB disk free, load $load on $(nproc) threads"
(( CHECK )) && exit 0

SUMMARY="$LOG_DIR/summary.tsv"
: > "$SUMMARY"
export THREADS MEM SCOPE LOG_DIR SUMMARY SLICE
run_one() {
  local line=$1 id cmd start rc
  id=${line%%$'\t'*}; cmd=${line#*$'\t'}
  [[ -z $id || $id == "$line" ]] && { echo "skipping malformed line: $line" >&2; return 0; }
  start=$(date +%s)
  if (( SCOPE )); then
    TM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
      systemd-run --user --scope --quiet --slice="$SLICE" -p MemoryMax="$MEM" -p MemorySwapMax=0 -- bash -c "$cmd" > "$LOG_DIR/$id.log" 2>&1
  else
    TM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS bash -c "$cmd" > "$LOG_DIR/$id.log" 2>&1
  fi
  rc=$?
  printf '%s\t%s\t%s\n' "$id" "$rc" "$(( $(date +%s) - start ))" >> "$SUMMARY"
  printf '[%s] %-60s exit %s\n' "$(date '+%T')" "$id" "$rc"
  return 0
}
export -f run_one

n=$(grep -c $'\t' "$JOBFILE")
echo "running $n jobs from $JOBFILE: $JOBS at a time, $MEM and $THREADS thread(s) each, logs in $LOG_DIR"
grep $'\t' "$JOBFILE" | xargs -d '\n' -P "$JOBS" -I{} bash -c 'run_one "$1"' _ {}

failed=$(awk -F'\t' '$2 != 0' "$SUMMARY" | wc -l)
echo "done: $((n - failed))/$n succeeded; summary in $SUMMARY"
if (( failed )); then awk -F'\t' '$2 != 0 {print "  FAILED " $1 " (exit " $2 ")"}' "$SUMMARY"; exit 1; fi
