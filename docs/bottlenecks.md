# Compute bottlenecks: MacroHFT and EarnHFT (before and after)

Measured on the BOX (AMD Ryzen 9 9950X3D, 16 cores / 32 threads, CPU-only torch 2.0.1, one thread per process)
with real Binance BTCUSDT klines and a synthetic 5-level order book (`box/bench/build_data.py`; Binance's free
archive has no order-book history). Timings depend on data shape and length, not on the values.

- **"Before":** the first BOX benchmark, from the original logs in `box/bench/logs/`.
- **"After":** `box/bench/box_bench.sh` on the same data sizes, after the fixes described in `MacroHFT/TRADEMASTER.md` and `EarnHFT/TRADEMASTER.md`.
- **Speed-ups change no results.** Actions, rewards and weights are bit for bit identical, checked by `box/bench/golden.py` and the tests in `box/bench/tests/`.
- **The bug fixes do change results,** and are listed separately in those files.

## MacroHFT (1-minute bars)

| Stage | Data | Before | After |
|---|---|---|---|
| Hyper-agent training (1 epoch) | 40,687 bars | 392.8 s, **9.6 ms/bar** | 34.5 s, **0.85 ms/bar** (11×) |
| Hyper-agent validation | 16,059 bars | 0.76 ms/bar | 0.29 ms/bar |
| Sub-agent training (1 epoch, 3 chunks × 4,320) | 12,960 bars | 0.81 ms/bar (4 threads) | 0.58 ms/bar (1 thread) |
| All six sub-agents, 1 epoch | | sequential | 8 s wall, in parallel |
| Decomposition | ~65-70k bars | 14.5 s | 0.7 s |

What changed:
- **Episodic memory query** (was ~54% of hyper-agent time): one vectorised kernel over the buffer instead of 4,320 Python calls per step.
- **Autograd:** 15 single-row forwards per step used to build graphs. They now run under `no_grad`, and the next step reuses the Q-values `q_estimate` just computed. The frozen sub-agents have `requires_grad_(False)`.
- **Environment:** numpy arrays extracted once, instead of `df.iloc` and column selection every step. An O(N²) end-of-episode loop is now `np.cumsum`.
- **Q-teacher:** vectorised, ~45× faster. `train.feather` is read once.
- **Rolling slope:** a 360-weight dot product instead of a filter-and-regression fit per row.

What remains: the networks themselves. A step runs 6 sub-agents plus the hyper-agent at batch size 1, which is about 70% of the remaining profile. Fusing the six sub-agents into one batched module would cut that further, but results would no longer be bit-identical.

## EarnHFT (1-second bars)

| Stage | Data | Before | After |
|---|---|---|---|
| Low-level training | 8 samples of a 14,400-step chunk (teacher + policy pass) | 181.7 s (~22.7 s/sample) | 7.2 s (~0.9 s/sample, 25×) |
| Market-dynamics labelling | 68,388-second valid set | **crashed** after 214 s | 158 s |
| Validation of one epoch × initial position | 68,388 steps over 336 segments | ~0.5 ms/step (estimated from the code) | ~5 s per job, 73 µs/step |
| Router training | 2 passes over 205,164 seconds | 140.6 s (1 pass) | 16.7 s (2 passes, ~37 µs per second-step) |
| Router test (valid + test) | 2 × 68,388 seconds | – | 7 s per epoch |

What changed:
- **Q-teacher:** vectorised and bit-identical. It was ~0.5 ms/row, computed twice per sample and 4× across betas; it is now 100-150× faster, and the policy pass reuses the teacher-pass environment.
- **Environment:** numpy arrays for the book and features. `cumsum` replaces the O(N²) balance loop, which cost ~11 min per router pass on a 2.4M-second train set.
- **Router:** loads the pool agents and train set once.
- **Inference:** runs under `no_grad`.
- **Threads:** capped before `import torch`.
- **Labelling:** the crash (a lost end-of-data sentinel) is fixed.

### Projected full runs on the BOX (upstream defaults, 1 thread per job)
- **`train-low`:** ~1.1 s per sample → ~4 min per beta at `num_sample=200`. The 4 betas run in parallel.
- **`valid-low`:** 250 jobs per beta × 73 µs per valid second. On a 0.8M-second valid set that is ~1 min per job, so ~1 h for 1,000 jobs at 16 in parallel. Serially it was an estimated ~115 h.
- **Router:** ~37 µs per train second → ~90 s per pass over 2.4M seconds, ~2.5 h for `num_sample=100`. It is one sequential process, so it is the critical path.
- **Labelling:** fastdtw, which is pure Python, now dominates. It is the next EarnHFT target; a C DTW would change the labels slightly.

What remains:
- The replay buffer stores a Python dict per transition: ~38 MB per sample, ~3.2 GB per training job at the default capacity. `sample()` also builds tensors from 512 dicts per update.
- A structured numpy buffer would cut both, but it changes the RNG use, so results would change.
- The roadmap's "precompute the router's pool actions" item is not done. The router is now ~90 s per pass, and batched forwards would not be bit-identical.

## Parallel scaling on the BOX (after)

N copies of a 1-epoch hyper-agent job (9,000 bars each, including ~2 s start-up), one thread per process:

| Processes | Wall | Aggregate bars/s | vs. 1 process |
|---|---|---|---|
| 1 | 5.5 s | 1,622 | 1.0× |
| 8 | 7.1 s | 10,173 | 6.3× |
| 16 | 7.4 s | 19,546 | 12.0× |
| 32 | 14.4 s | 19,964 | 12.3× |

- **Scaling stops at 16 processes**, the physical core count. SMT siblings give nothing for this workload, so run `box/runjobs.sh` at `-j 16` or less.
- **Memory:** the hyper-agent peaks at 0.5 GB and the router at 1.1 GB. EarnHFT low-level training grows to ~3.2 GB (see `box/README.md`). A per-job `MemoryMax` is enforced: an over-limit job is killed alone.
- **CPU beats GPU for these workloads.** One action per bar with tiny tensors means host↔device copies dominate. Laptop benchmark (i7-11800H vs RTX 3070): action selection took 152 µs on CPU vs 578 µs on GPU.

## Timeframe

Per-bar cost doesn't depend on bar size, so compute scales linearly with the number of bars (1 year ≈ 31.5M 1-second bars, 525,600 1-minute bars, 35,040 15-minute bars or 8,760 1-hour bars). The window constants count bars. They are configurable now:
- MacroHFT: `MACRO_CHUNK_SIZE` and `MACRO_CONTEXT_WINDOW` for `decompose`; `--context_window` and `--memory_capacity` for `train-high`.
- EarnHFT: `--chunk_length` and `--future_sight`.

The `*_trend_60` context features are computed in the data files.
