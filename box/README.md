# box/: CPU compute tooling (the BOX) and benchmarks

The BOX is a borrowed 16-core / 32-thread Ryzen 9 9950X3D reached as `ssh chad-box`. TradeMaster lives in
`/home/chad/TradeMaster` there, with a CPU-only `.venv-hft` (`torch 2.0.1+cpu`, from `requirements-hft-cpu.txt`).
MacroHFT and EarnHFT run one action per bar through tiny networks, so they are CPU- and Python-bound. A single
process gains almost nothing from extra threads, but many 1-thread processes scale nearly linearly. Everything
here is built around that.

**Host rules** (see the root `CLAUDE.md`):
- Stay at or below 48 GB of RAM in total, and keep at least 20 GiB of disk free.
- Never touch the owner's files or processes.
- Copying files to the host needs the user's approval.

## Running jobs: `runjobs.sh`

```bash
EMIT_JOBS=1 EarnHFT/run.sh valid-low ETHUSDT > jobs.tsv    # any stage of either run.sh: list its jobs
box/runjobs.sh -j 16 -m 1500M jobs.tsv                      # run them, 16 at a time, 1.5 GB cap each
```

- Each job runs in its own `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`, so a runaway job is killed alone.
- Each job gets `TM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to `-t` (default 1).
- Logs go to `<jobfile>_logs/<job id>.log`, with `summary.tsv` holding the id, exit code and seconds.
- A failed job does not stop the others. The script exits 1 at the end if any job failed.
- It refuses to start if `-j × -m` exceeds `BOX_MEM_BUDGET` (48G) or less than 20 GiB of disk is free.

Suggested sizes, measured on the BOX. It has **16 physical cores**, and SMT siblings add nothing: 16 one-thread
hyper-agent jobs give about 12× the throughput of one, and 32 give no more than 16. Keep `-j` at 16 or below for CPU-bound work
(fewer if the owner is busy: check `uptime`).

| Workload | Peak RSS (measured) | `-j` | `-m` |
|---|---|---|---|
| MacroHFT sub-agents (`train-low`) | ~0.4 GB | 6 (all at once) | 1500M |
| MacroHFT hyper-agent (`train-high`), extra seeds/exps side by side | 0.5 GB | up to 16 | 1500M |
| EarnHFT `train-low` (one job per beta or seed) | 0.5 GB + ~38 MB per sample until the 1e6 replay buffers fill (~70 samples): ~3.2 GB at the default 200 | 12 | 4G |
| EarnHFT `train-high` (router) | 1.1 GB | 1 per seed | 2G |
| EarnHFT `valid-low` / `test-high` | < 1 GB | 16 | 1500M |
| EarnHFT `label` | < 1 GB | 1 | 4G |

Sync code with rsync: send only the files git tracks, plus new ones it doesn't ignore, under the HFT paths
(environments, data and results are git-ignored, so they never go). Never use `--delete`:

```bash
git ls-files -co --exclude-standard -- MacroHFT EarnHFT box requirements-hft*.txt CLAUDE.md > /tmp/sync.list
rsync -az --files-from=/tmp/sync.list ./ chad-box:/home/chad/TradeMaster/     # add --dry-run --itemize-changes to preview
```

Pull results back with `rsync -az chad-box:/home/chad/TradeMaster/<project>/result… ./…`.

## Benchmarks and checks: `bench/`

| File | What |
|---|---|
| `bench/build_data.py` | Builds benchmark data from real Binance BTCUSDT klines (data.binance.vision) with a synthetic 5-level book: `macro <interval> <start> <train> <val> <test> [PAIR]` writes `MacroHFT/data/<PAIR>/df_*.feather`; `earn <start> <days>` writes `EarnHFT/EarnHFT_Algorithm/data/BTCUSDT/df.feather` plus the feature lists. Raw zips are cached in `bench/raw/` (git-ignored). |
| `bench/golden.py` | Golden-output check. `prepare` freezes small `BTCGOLD` datasets. `run <name> [macro\|earn\|all]` runs every pipeline stage single-threaded and fingerprints the saved actions, rewards, model weights and printed results. `compare <a> <b>` diffs two runs: bit-exact, except that reported metrics use a relative 1e-12. Use it to prove that a performance change leaves results unchanged. Only compare runs from the same machine: the torch CPU kernels round differently on the BOX's AMD Zen 5 and the laptop's Intel CPU, so the weights differ in the last bits. |
| `bench/golden_earn.py` | The EarnHFT stages for `golden.py`. |
| `bench/tests/test_macro_memory.py` | The vectorised MacroHFT episodic-memory query equals the upstream loop, bit for bit. |
| `bench/tests/test_q_tables.py` | The vectorised Q-teacher tables (both projects) equal the upstream loops, bit for bit. |
| `bench/tests/test_macro_decomposition.py` | The dot-product rolling slope equals upstream `rolling().apply(...)` to float rounding. |
| `bench/ts.py` | Prefixes stdin lines with a timestamp (used for the timing logs). |
| `bench/logs/` | The original BOX benchmark logs, before any fixes. |

```bash
.venv-hft/bin/python box/bench/build_data.py macro 1m 2024-01-01 30 12 9 BTCUSDT
.venv-hft/bin/python box/bench/build_data.py earn 2024-03-01 4
.venv-hft/bin/python box/bench/golden.py prepare
.venv-hft/bin/python box/bench/golden.py run before && <change code> && .venv-hft/bin/python box/bench/golden.py run after
.venv-hft/bin/python box/bench/golden.py compare before after
for t in box/bench/tests/test_*.py; do .venv-hft/bin/python $t; done
```

`box/init_box.sh`, which older notes mention, was never committed and is not needed. `ssh`/`rsync` access and the
host's `.venv-hft` already exist, and `runjobs.sh` enforces the host rules on every run.
