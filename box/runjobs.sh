#!/usr/bin/env bash
# Run independent CPU jobs in parallel under a per-job memory cap (TradeMaster on the BOX).
#
#   box/runjobs.sh [-j JOBS] [-m MEM_PER_JOB] [-l LOG_DIR] [-t THREADS] [--no-scope] JOBFILE
#
# JOBFILE: one job per line, "<job id>\t<shell command>" (what `EMIT_JOBS=1 <project>/run.sh <stage>` prints).
# Every job runs with TM_THREADS/OMP/MKL = THREADS (default 1) inside its own
#   systemd-run --user --scope -p MemoryMax=MEM -p MemorySwapMax=0
# so one runaway job is killed alone. A failed job does not stop the others; each job's output goes to
# LOG_DIR/<job id>.log, results to LOG_DIR/summary.tsv (id, exit code, seconds), and the script exits 1
# if any job failed.
#
# BOX rules enforced here: JOBS x MEM_PER_JOB <= BOX_MEM_BUDGET (default 48G) and >= 20 GiB free disk.
set -uo pipefail

JOBS=8; MEM=1500M; LOG_DIR=""; THREADS=1; SCOPE=1
BUDGET="${BOX_MEM_BUDGET:-48G}"; MIN_FREE_GIB="${BOX_MIN_FREE_GIB:-20}"
while [[ $# -gt 0 ]]; do
  case $1 in
    -j) JOBS=$2; shift 2 ;;
    -m) MEM=$2; shift 2 ;;
    -l) LOG_DIR=$2; shift 2 ;;
    -t) THREADS=$2; shift 2 ;;
    --no-scope) SCOPE=0; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
JOBFILE=${1:?usage: runjobs.sh [-j N] [-m MEM] [-l LOG_DIR] [-t THREADS] [--no-scope] JOBFILE}
[[ -f $JOBFILE ]] || { echo "no such job file: $JOBFILE" >&2; exit 2; }
LOG_DIR=${LOG_DIR:-$(dirname "$JOBFILE")/$(basename "$JOBFILE" .tsv)_logs}
mkdir -p "$LOG_DIR"

bytes() { numfmt --from=iec "${1%B}"; }
if (( JOBS * $(bytes "$MEM") > $(bytes "$BUDGET") )); then
  echo "refusing: $JOBS jobs x $MEM > memory budget $BUDGET (lower -j or -m, or set BOX_MEM_BUDGET)" >&2; exit 2
fi
free_gib=$(( $(df --output=avail -k "$LOG_DIR" | tail -1) / 1048576 ))
if (( free_gib < MIN_FREE_GIB )); then
  echo "refusing: only ${free_gib} GiB free on $(df --output=target "$LOG_DIR" | tail -1) (need >= $MIN_FREE_GIB)" >&2; exit 2
fi
if (( SCOPE )) && ! systemd-run --user --scope --quiet -p MemoryMax=64M true 2>/dev/null; then
  echo "systemd-run --user --scope is unavailable here; rerun with --no-scope (no per-job memory cap)" >&2; exit 2
fi

SUMMARY="$LOG_DIR/summary.tsv"
: > "$SUMMARY"
export THREADS MEM SCOPE LOG_DIR SUMMARY
run_one() {
  local line=$1 id cmd start rc
  id=${line%%$'\t'*}; cmd=${line#*$'\t'}
  [[ -z $id || $id == "$line" ]] && { echo "skipping malformed line: $line" >&2; return 0; }
  start=$(date +%s)
  if (( SCOPE )); then
    TM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
      systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 -- bash -c "$cmd" > "$LOG_DIR/$id.log" 2>&1
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
