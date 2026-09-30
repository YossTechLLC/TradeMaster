# MacroHFT in TradeMaster

**Paper:** *MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading* (KDD 2024), [PDF](https://personal.ntu.edu.sg/boan/papers/KDD24_MacroHFT.pdf) · [arXiv 2406.14537](https://arxiv.org/abs/2406.14537) (local copy: `articles/MacroHFT.pdf`)
**Upstream:** https://github.com/ZONG0004/MacroHFT @ `31e5ef41f93b2aea6e63e6ae2267675c997a8e54` (2025-01-09), vendored without history.
The upstream `readme.md` is kept. This file explains how the project is wired into TradeMaster.

## What it does

MacroHFT trades crypto at minute level (ETHUSDT in the demo) in two phases.

| Phase (paper) | Idea | Code |
| --- | --- | --- |
| Market decomposition | Cut the data into chunks of 4320 minutes (3 days). Label each chunk by **trend** (slope of a low-pass-filtered linear fit) and by **volatility**, split at the 5/35/65/95% quantiles into five classes, of which the middle three train the sub-agents. Validation and test sets reuse the training thresholds. | `preprocess/decomposition.py` |
| I. Sub-agents | One dueling DDQN per market type: bear, flat, bull, stable, medium, volatile. Each has a **conditional adapter**: context features and the current position shift and scale the hidden state through LayerNorm modulation. A DP "optimal action" demonstration adds a KL term to the loss. | `RL/agent/low_level.py`, `model/net.py` (`subagent`, `modulate`), `tools/demonstration.py`, `env/low_level_env.py` |
| II. Hyper-agent | Mixes the six sub-agents' Q-values with softmax weights conditioned on market context. An **episodic memory** (kernel-weighted lookup of similar past states) stabilises it in extreme markets. | `RL/agent/high_level.py`, `model/net.py` (`hyperagent`, `calculate_q`), `RL/util/memory.py`, `env/high_level_env.py` |

Features: `data/feature_list/single_features.npy` (36 single-state features) and `trend_features.npy` (9 context features). Both ship with the repo and are loaded when the env modules are imported.

## Layout

```
MacroHFT/
├── run.sh                        TradeMaster launcher (one device, .venv-hft, foreground, logs/; EMIT_JOBS=1 lists jobs)
├── data/feature_list/            feature lists (shipped)
├── data/<PAIR>/                  df_{train,val,test}.feather (ETHUSDT from Google Drive) + decomposition output (git-ignored)
├── result/low_level/ETHUSDT/best_model/{slope,vol}/{1,2,3}/best_model.pkl   pretrained sub-agents (shipped, tracked)
├── result/high_level/<PAIR>/<exp>/seed_<n>/   hyper-agent runs: epochs, best_model.pkl, test/ (git-ignored)
└── scripts/                      upstream multi-GPU scripts (reference only)
```

The code imports itself as the `MacroHFT` package (`from MacroHFT.model.net import *`), with the TradeMaster root on `sys.path`. **Keep the folder named `MacroHFT`.**

## Running it

Environment: the shared `../.venv-hft` (Python 3.10, torch 2.0.1). On a CPU-only machine (the BOX) install
`requirements-hft-cpu.txt` instead; see the root README. The models are tiny MLPs run one bar at a time, so the
CPU is faster than a GPU here. Use `DEVICE=cpu`.

1. Download the `ETHUSDT` folder from the Google Drive link in `readme.md` into `MacroHFT/data/`, so that `data/ETHUSDT/df_train.feather`, `df_val.feather` and `df_test.feather` exist. For another pair, put the same three files under `data/<PAIR>/` and set `DATASET=<PAIR>`. `box/bench/build_data.py macro ...` builds them from Binance klines, with a synthetic order book.
2. Run the stages:

```bash
./run.sh help
./run.sh decompose                    # Step 1: chunk and label -> data/$DATASET/{train,val,test,whole}/
./run.sh train-low                    # Step 2 (optional for ETHUSDT): train all six sub-agents one after another
./run.sh train-low slope label_1 1    #   ...or one sub-agent: <clf> <label> <alpha>
./run.sh train-high                   # Step 3: hyper-agent -> result/high_level/$DATASET/<exp>/seed_<n>/
```

Environment overrides: `DATASET` (default ETHUSDT), `DEVICE` (default cuda:0), `TM_THREADS` (threads per process, default 1). Anything after `--` is passed to Python.

The six pretrained ETHUSDT sub-agents ship in `result/low_level/ETHUSDT/best_model/`. For ETHUSDT you can go straight from `decompose` to `train-high`. Other pairs need `train-low` first, or `-- --subagent_path DIR`.

**Parallel (BOX):** `EMIT_JOBS=1 ./run.sh <stage>` prints the stage's jobs instead of running them. `box/runjobs.sh` then runs them side by side, one thread each, with a memory cap per job:

```bash
EMIT_JOBS=1 DATASET=BTCUSDT DEVICE=cpu ./run.sh train-low > /tmp/low.tsv
../box/runjobs.sh -j 6 -m 1500M /tmp/low.tsv          # all six sub-agents at once
```

The hyper-agent is one sequential process. Use the other cores for extra seeds or experiments: `-- --seed N --exp NAME` writes to its own `result/high_level/<DATASET>/<exp>/seed_<n>/`.

Bar size: the constants count bars (upstream uses 1-minute bars). For coarser bars, set `MACRO_CHUNK_SIZE` (default 4320) and `MACRO_CONTEXT_WINDOW` (default 360) for `decompose`, and pass `-- --context_window W --memory_capacity N` to `train-high`. The `*_trend_60` context features come from the data files, not from this code.

## Adaptations vs upstream

Every change is marked `# [TradeMaster]` in the source.
- **perf** changes leave results unchanged. They are checked with `box/bench/golden.py`: actions, rewards and model weights are bit for bit identical, and reported metrics agree to a relative 1e-12.
- **semantic** changes are bug fixes that change results.

`run.sh` runs every job on one device in the foreground. Upstream used `cuda:0-3` with `nohup … &` and expected `logs/…` to already exist.

### Crash and pipeline fixes
1. The end-of-training `torch.save(best_model.state_dict(), …)` called `.state_dict()` on something that was already a state dict, so it raised `AttributeError` after all epochs finished. The best weights are now also deep-copied when selected; before, they were a live reference to the final weights.
2. `low_level.py` saved its best model to `result/low_level/<ds>/<clf>/label_<n>/best_model.pkl`, a directory that is never created, and not where `high_level.py` loads sub-agents from. It now saves to `result/low_level/<ds>/best_model/<clf>/<n>/best_model.pkl`, so Step 2 feeds Step 3 directly. For ETHUSDT, retraining **overwrites the shipped checkpoints** (`git checkout MacroHFT/result` restores them).
3. `high_level.py` `train()` passed the saved `best_model.pkl` file to `test_cluster()`, which appended `/trained_model.pkl` to it, so the final test crashed with `NotADirectoryError`. `test_cluster()` now accepts either a file or an epoch directory.
4. **NaN validation:** an agent that never buys has `required_money == 0`, so its return rate is `nan`. `nan` never beat the best score, so `best_model` stayed `None` and the final test crashed on `load_state_dict(None)` (reproduced on BTCUSDT). The hyper-agent now scores a non-finite return rate as 0. The sub-agents, when no validation chunk carries their label, keep the latest epoch instead of saving `None`.
5. The dataset is a parameter (`DATASET`, `decomposition.py <PAIR>`, `--subagent_path`); ETHUSDT was hard-coded. The hyper-agent's best model and test output go under `<exp>/seed_<n>/`, so different runs no longer overwrite each other.
6. `decomposition.py` uses open-ended outer bins (`±inf`). The previous `±100` (slope) and `0/1` (volatility) bins gave NaN labels, and a crash, for a pair with larger values.
7. `run.sh train-low` with 1 or 2 arguments used to train all six sub-agents; it is now an error. `train-high` checks its input files.

### Performance (perf, golden-checked)
Measured on 2 epochs × 6000 bars of the hyper-agent: **212 s → 29 s** (laptop i7-11800H, 1 thread). Sub-agent job: 34 s → 13 s.
- `RL/util/memory.py` `query()`: one vectorised kernel over the buffer instead of 4320 Python calls per step. This was ~54% of hyper-agent time.
- `high_level.py`: `act`/`q_estimate` run without autograd. The next step's `act(s_)` reuses the Q-values `q_estimate(s_)` just computed when no update ran in between. The frozen sub-agents have `requires_grad_(False)`, and the target side of `update()` runs under `no_grad`. `low_level.py` has the same `no_grad` changes.
- `env/*_env.py`: feature, trend, context and close columns are extracted to numpy once; `step()` no longer does `df.iloc` or column selection. `get_final_return_rate` uses `np.cumsum` instead of an O(N²) loop of prefix sums.
- `tools/demonstration.py`: the Q-teacher table is vectorised, bit-identical and ~45× faster. `train.feather` is read once, not every epoch.
- Thread caps (`TM_THREADS`, default 1) are applied before `import torch`, plus `torch.set_num_threads`. Upstream set them after the import, where they had no effect.
- `decomposition.py`: the rolling `slope_360` is a fixed 360-weight dot product instead of one butter, filtfilt and LinearRegression fit per row. It is 9× faster and equal to within 1.5e-13 of the slope scale.

### Semantic fixes (results change)
- `model/net.py` dueling head: `advantage.mean(dim=-1, keepdim=True)`. The upstream `advantage.mean()` averaged over the whole batch, so the Q-values of a 512-row batch in `update()` were shifted differently from the batch-1 Q-values in `act()`: up to 0.54 on a batch of 4. At batch size 1 both forms are identical, so **the shipped sub-agent checkpoints give the same actions as before** and stay usable. What changes is how the hyper-agent's training batches see them.
- The Q-teacher uses the environment's commission (`transcation_cost`, 0.0002). It was hard-coded to 0.001.
- `high_level_env.py` pays for the initial position, as `low_level_env.py` does. A random `initial_action=1` in training used to be a free position, which made `required_money` 0 or negative.
- The hyper-agent standardises the `slope_360` and `vol_360` context features with train-split statistics, saved to `<run>/clf_normalisation.yaml`. They went unnormalised into `fc2` (a raw price slope next to a ~1e-3 volatility).
- `low_level.py` and `high_level.py` methods read the module-global `args`, which only worked when run as `__main__`. They now use instance attributes.

### Known, not changed
- The market-type labels use quantiles `[0, .05, .35, .65, .95, 1]`, i.e. five classes, not terciles. The sub-agents train on classes 1-3, so the most extreme 5% of train chunks at each end are unused. Validation and test fold 0→1 and 4→3.
- Execution is at the previous bar's close with zero latency and zero slippage.
- Validation and test run one episode from a flat position.
