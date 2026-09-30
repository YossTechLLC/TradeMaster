# box/: CPU compute tooling (the BOX) and benchmarks

The BOX is a borrowed 16-core / 32-thread Ryzen 9 9950X3D reached as `ssh chad-box`. TradeMaster lives in
`/home/chad/TradeMaster` there, with a CPU-only `.venv-hft` (`torch 2.0.1+cpu`, from `requirements-hft-cpu.txt`).
MacroHFT and EarnHFT run one action per bar through tiny networks, so they are CPU- and Python-bound. A single
process gains almost nothing from extra threads, but many 1-thread processes scale nearly linearly. Everything
here is built around that.

**Host rules** (see the root `CLAUDE.md`):
- Stay within our allocation of 16 CPUs / 24 GB RAM (the host is shared with the SIMONS agent), and keep at least 20 GiB of disk free.
- Never touch the owner's files or processes.
- Copying files to the host needs the user's approval.

## Running jobs: `runjobs.sh`

```bash
EMIT_JOBS=1 EarnHFT/run.sh valid-low ETHUSDT > jobs.tsv    # any stage of either run.sh: list its jobs
box/runjobs.sh -j 16 -m 1500M jobs.tsv                      # run them, 16 at a time, 1.5 GB cap each (24 GB)
```

- Each job runs in its own `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`, so a runaway job is killed alone.
- Each job gets `TM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` set to `-t` (default 1).
- Logs go to `<jobfile>_logs/<job id>.log`, with `summary.tsv` holding the id, exit code and seconds.
- A failed job does not stop the others. The script exits 1 at the end if any job failed.
- It refuses to start if the plan exceeds our 16 CPU / 24 GB allocation, the host lacks that memory right now, or less than 20 GiB of disk is free (see below).

**Shared host: our allocation is 16 CPUs / 24 GB.** Another agent runs SIMONS compute on the BOX under its own
16 CPUs / 24 GB. `runjobs.sh` runs every job of ours inside one systemd slice (`tmhft.slice`) that the kernel caps
at 24 GB and 16 CPUs in total, even across several `runjobs` invocations at once. It refuses plans where
`-j × -m` > 24G or `-j × -t` > 16, or where the host doesn't currently have the planned memory available.
`box/runjobs.sh --check -j N -m MEM` runs only these checks.

The BOX has **16 physical cores**, and SMT siblings add nothing: 16 one-thread hyper-agent jobs give about 12× the
throughput of one, and 32 give no more than 16. Our 16 CPUs are therefore a quota, not pinned cores; expect lower
per-job speed while the other agent is busy.

Sizes within the 24 GB allocation (peak RSS measured on the BOX; the replay buffers grow as they fill):

| Workload | Peak RSS | `-j` | `-m` | Total |
|---|---|---|---|---|
| MacroHFT `decompose` | < 1 GB | 8 | 2G | 16 GB |
| MacroHFT sub-agents (`train-low`) | ~0.4 GB, up to ~1.2 GB with a full buffer | 16 | 1500M | 24 GB |
| MacroHFT hyper-agent (`train-high`), seeds/exps side by side | 0.5 GB, ~1.4 GB with a full 1e6 buffer | 12 | 2G | 24 GB |
| EarnHFT `train-low` | ~3.2 GB at the default 200 samples | 6 | 4G | 24 GB |
| EarnHFT `train-high` (router) | 1.1 GB | 1 per seed | 2G | |
| EarnHFT `valid-low` / `test-high` | < 1 GB | 16 | 1500M | 24 GB |
| EarnHFT `label` | < 1 GB | 1 | 4G | |

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
| `bench/tests/test_profiles.py` | MacroHFT input profiles: every feature is causal (identical values when later bars are cut off), and every built dataset is finite, scaled and accepted by MacroHFT's env and networks. |
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
