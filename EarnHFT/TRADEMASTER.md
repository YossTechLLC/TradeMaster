# EarnHFT in TradeMaster

**Paper:** *EarnHFT: Efficient Hierarchical Reinforcement Learning for High Frequency Trading* (AAAI 2024), [PDF](https://personal.ntu.edu.sg/boan/papers/AAAI24_EarnHFT.pdf) · [arXiv 2309.12891](https://arxiv.org/abs/2309.12891) (local copy: `articles/EarnHFT.pdf`)
**Upstream:** https://github.com/TradeMaster-NTU/EarnHFT @ `0e1e11a6d9aff70efb1807baa3416429568deb31` (2024-07-19), vendored without history.
The upstream `README.md` files are kept unchanged. This file explains how the project is wired into TradeMaster.

## What it does

EarnHFT treats second-level crypto trading as a two-level MDP: every **minute**, a router picks one agent from a small pool, and that agent sets a target position every **second**.

| Stage (paper) | Idea | Code |
| --- | --- | --- |
| I. Q-teacher | Dynamic programming with future prices gives the optimal action values. These act as a KL regulariser on the DDQN, which speeds up training on very long trajectories. | `EarnHFT_Algorithm/tool/demonstration.py` (`make_q_table_reward`), used by `RL/agent/low_level/ddqn_pes_risk_aware.py` (`demonstration_loss`) |
| II. Diverse agent pool | Train second-level agents with different return-rate sampling preferences `beta`. Label valid-set market segments (DTW slice-and-merge), then keep the best agent per market category and initial position. | Chunk sampling: `RL/util/episode_selector.py`. Segment labels: `tool/slice_model.py`. Evaluation: `RL/agent/low_level/test_ddqn.py`. Selection: `analysis/pick_agent/pick_agent_position.py` |
| III. Router | Minute-level DQN whose action is "which pool agent to run for the next minute" | `RL/agent/high_level/dqn_position.py`, `env/high_level_env.py`, `model/net.py` (`Qnet_high_level_position`) |
| Environment | 5-level LOB market-order execution with commission. Only long positions. | `env/low_level_env.py`, `env/high_level_env.py` |
| Baselines | DQN, CDQN-RP, DRA (LSTM+PPO), PPO, rule-based | `RL/agent/base/`, `script/base/` |

The state uses two feature lists chosen by IC analysis: second-level LOB/OHLC features (`second_feature.npy`) and minute-level OHLC features (`minitue_feature.npy`). The paper reports 54 and 19 features; your data may produce different counts.

## Layout

```
EarnHFT/
├── run.sh                 TradeMaster launcher (.venv-hft, foreground, logs; EMIT_JOBS=1 lists jobs)
├── data_preprocess/       Tardis download -> clean -> features -> merge -> IC feature selection
└── EarnHFT_Algorithm/     RL code; every script expects to run from this directory
    ├── data/<PAIR>/       df.feather, then train/ valid/ test splits (generated, git-ignored)
    ├── data/feature/<PAIR>/  second_feature.npy, minitue_feature.npy per pair (generated, loaded at import time)
    ├── result_risk/<PAIR>/  low-level agents, potential_model/ pool, high_level/ router (git-ignored)
    └── script/<PAIR>/     upstream multi-GPU scripts (reference only)
```

## Running it

Environment: the shared `../.venv-hft` (Python 3.10, torch 2.0.1). On a CPU-only machine (the BOX) install
`requirements-hft-cpu.txt` instead; see the root README. Every stage is CPU-bound, and the scripts use the
GPU only if one is visible (`CUDA_VISIBLE_DEVICES=` hides it).

```bash
./run.sh help
./run.sh download   ETHUSDT 2022-05-01 2022-06-15   # needs a Tardis API key + pip install tardis-dev nest_asyncio
./run.sh preprocess ETHUSDT 2022-05-01 2022-06-15   # installs df.feather + data/feature/ETHUSDT/*.npy
./run.sh split      ETHUSDT
./run.sh label      ETHUSDT
./run.sh train-low  ETHUSDT            # betas -10 -90 30 100, one after another (or: train-low ETHUSDT 30)
./run.sh valid-low  ETHUSDT            # every epoch (up to MAX_VALID_EPOCHS) x 5 initial positions
./run.sh pick       ETHUSDT            # agent pool -> result_risk/ETHUSDT/potential_model
./run.sh train-high ETHUSDT            # router
./run.sh test-high  ETHUSDT            # every router epoch on valid + test (or one: test-high ETHUSDT <epoch dir>)
./run.sh pick-high  ETHUSDT            # choose the router epoch on valid, report its test return
```

Anything after `--` is passed to the Python script, for example `./run.sh train-low ETHUSDT 30 -- --epoch_number 5 --num_sample 40`. Environment overrides: `SEED`, `RISK_BOND`, `BETAS`, `MAX_HOLD`, `TC`, `MAX_VALID_EPOCHS`, `MAX_TEST_EPOCHS`, and `TM_THREADS` (threads per process, default 1).

**Parallel (BOX):** `EMIT_JOBS=1 ./run.sh <stage> ...` prints the stage's jobs instead of running them. `box/runjobs.sh` runs them side by side with a memory cap per job. `train-low` gives one job per beta, `valid-low` one per epoch × initial position (up to 1000), and `test-high` one per router epoch:

```bash
EMIT_JOBS=1 ./run.sh valid-low ETHUSDT > /tmp/valid.tsv
../box/runjobs.sh -j 16 -m 1500M /tmp/valid.tsv
```

Bar size: EarnHFT is built around 1-second bars with a router step per minute (`timestamp.second == 59`). The chunk sizes are flags: `split --chunk_length/--future_sight`, `train-low -- --chunk_length`. The labeller has `--min_length_limit` and `--max_length_expectation`.

### Data requirements

- **Tardis.dev** (`book_snapshot_5` + `trades`) is the only supported source, and full date ranges need a paid API key. Put the key in `data_preprocess/download_code/download.py` (`api_key=`). `tardis-dev` is left out of `requirements-hft.txt` on purpose because it is only needed for downloading.
- Nothing ships with the repo. The RL scripts fail at import time with `FileNotFoundError: data/feature/second_feature.npy` until `preprocess` has run.
- If you bring your own LOB data, match the columns described in `data_preprocess/README.md` and write `EarnHFT_Algorithm/data/<PAIR>/df.feather` and the two feature `.npy` lists (in `data/feature/<PAIR>/`) yourself. `box/bench/build_data.py earn ...` does this from Binance 1-second klines, with a synthetic order book.
- The PES chunk sampler fits a KDE with 5-fold cross-validation, so `train-low` needs at least 5 training chunks (5 × 4 hours by default).

## Adaptations vs upstream

Every change is marked `# [TradeMaster]` in the source.
- **perf** changes leave results unchanged. They are checked with `box/bench/golden.py`, which runs every stage on a small dataset: actions, rewards and model weights are bit for bit identical, and reported metrics agree to a relative 1e-12.
- **semantic** changes are bug fixes that change results.

`run.sh` runs jobs one after another in the foreground, or in parallel through `EMIT_JOBS` + `box/runjobs.sh`. Upstream scripts spread jobs across GPUs 0–3 with `nohup … &`. Upstream `script/<PAIR>/high_level/pick.sh` calls `analysis/pick_agent_position.py`, which doesn't exist; `run.sh` uses `analysis/pick_agent/pick_agent_position.py`. Per-pair `--max_holding_number` and `--transcation_cost` follow the upstream scripts (ETHUSDT 0.1, GALAUSDT 4000, BTCUSDT 0.01, BTCTUSD 0.01 with zero fee). `run.sh` passes them to every stage, where upstream `valid_multi.sh` left them at their defaults.

### Crash and pipeline fixes
1. **IC feature selection leaked the future return:** `calculate_ic.py` wrote `df["return"]` (the next-period price change) into the shared frame. The period-60 pass then selected `return` itself as a minute feature, so `train-high` failed with `KeyError: 'return'` on real data. It now works on a copy and only considers numeric columns.
2. **`label` crashed** with `ValueError: Length of values (67616) does not match length of index (68388)` (seen on the BOX, reproduced locally). When a segment's right neighbour was the end-of-data sentinel, the DTW distance was `nan`. `left < nan` is False, so the segment was merged *into the sentinel* and the end index was lost. The sentinel is now never merged into, and `nan` distances count as infinite. Equal-length neighbours, which gave `mean([])` = nan and never merged, now get one DTW comparison.
3. `slice_model.py`: re-running `label` failed on `os.makedirs` (it now replaces `data/<PAIR>/valid/`). The last segment of the validation set was never written. Segment boundaries are found with one vectorised comparison.
4. `pick_agent_position.py`:
   - It only reads `beta_*` runs; `high_level/` and old folders used to crash it.
   - It skips epochs that `valid-low` didn't validate.
   - If a market type has no validation segment, it falls back to the best agent over all types, instead of crashing or silently picking the first key.
   - It loads with `map_location="cpu"`, and takes `--seed`.
5. The router's agent-pool paths were hard-coded for four pairs. They are now built for any pair from the `pick` output layout, with a clear error if the pool is missing.
6. `ddqn_pes_risk_aware.py`:
   - `--chunk_length` and `--epoch_number` (samples per checkpoint) are now flags; they were hard-coded to 14400 and 4. The `--epoch_number` example above used to be rejected.
   - Chunk discovery is `df_<i>.feather` files longer than one episode, instead of "every file in `train/` but one".
7. `run.sh`:
   - `SEED` and `RISK_BOND` are variables; they were hard-coded in result paths, so overriding `--seed` broke `valid-low` and `test-high`.
   - `test-high` tests every saved router epoch; upstream stopped at 80 of 100.
   - Feature lists are stored per pair (`data/feature/<PAIR>/`, selected through `EARNHFT_FEATURE_DIR`). They were one shared pair of files, so the last preprocessed pair won. `data/feature/` is still written for the upstream baselines.
8. `env/low_level_env.py` computed the first action mask from the *previous* position, before setting the initial one, so validation from positions 1-4 started with a wrong mask. `env/high_level_env.py` `reset()` now clears every history list; they used to grow by one entry per second across router passes, which is gigabytes.
9. `create_feature.py`: `df.drop(columns=["max_oc_*", "min_oc_*"])` discarded its result, so these price levels stayed in as candidate features.

### Performance (perf, golden-checked)
Golden run of every stage (laptop, 1 thread): **420 s → 88 s**. `train-low` is 95 s → 7 s, the router 80 s → 10 s, and `valid-low` 128 s → 55 s.
- `tool/demonstration.py`: the Q-teacher table is vectorised over rows, bit-identical and 100-150× faster. It was 0.5 ms per row, twice per training sample. The policy pass now reuses the teacher-pass environment instead of rebuilding it (and its Q-table) on the same chunk.
- `env/low_level_env.py`: features, the 5-level book and the level 1-4 size sums are extracted to numpy once. `step()` no longer does `df.iloc` or column selection; `env.data` remains available as a property. `get_final_return_rate` uses `np.cumsum` instead of an O(N²) loop, which cost ~11 min per router pass over a 2.4M-second train set.
- `env/high_level_env.py`: the minute-level state and the "last second of the minute" flags are pre-extracted. Pool agents run under `no_grad`.
- The router reads `train.feather` and loads the 25 pool agents once, not every pass. Low-level startup reads only the two columns it needs from each chunk. `act`/`act_test` run without autograd everywhere.
- Thread caps (`TM_THREADS`, default 1) are applied before `import torch`, plus `torch.set_num_threads`. Upstream set them after the import, where they had no effect.
- `create_feature.py`: `rolling(...).apply(np.argmax, raw=True)`, each computed once (they were computed twice, on a Series per window). `concat_clean.py` uses the C CSV parser with `float_precision="round_trip"` instead of `engine='python'`.

### Semantic fixes (results change)
- Router TD target: the next state's Q-values are now taken at the next position (`info_["previous_action"]`); upstream used the position before the step. Terminal transitions are now stored (`dones` was always 0), and the environment returns the router state on the terminal step.
- Position → level index: `int()` of the float ratio truncated `0.9999999` to 0, e.g. ETHUSDT 0.1 going 4 → 1 left `0.024999999999999994`. The router then used the agent pool of the wrong position, and the action mask lost a level. A `1e-9` tolerance fixes this. Off-grid positions from partial fills still map to the level below.
- Low-level DDQN: the TD target includes `gamma` (identical at the default `gamma=1`). Epsilon no longer decays during the teacher pass, where it isn't used and where it made exploration decay twice as fast as `--epsilon_step`. `ada` and `lr` still decay there, because updates happen in both passes.
- IC feature selection uses the training part of `df.feather` only (`--train_fraction 0.6`, matching `split_data.py`). It used the whole file, including the valid and test periods. Features are ordered deterministically.
- `analysis/pick_agent/high_level_analysis.py` (`run.sh pick-high`) chooses the router epoch on the **valid** set. It chose on test.
- `create_feature.py`: `ma_<w>_m` is `(mean - close) / std`, as `ma_<w>_s` is. It was `(std - close) / std`.

### Known, not changed
- The teacher's order-book walk uses `<=` where the environment uses `<`. Both give the same cash, so this is not a bug.
- Only levels 1-4 of the book are ever traded: reaching level 5 with volume left is punished or left unfilled, which is consistent with the action mask.
- On a partial fill, the environment reports the *intended* action as `previous_action`.
- IC selection keeps raw price-level columns (wap, vwap, OHLC) if they correlate.
- Replay buffers store Python dicts per transition: about 2-3 GB per low-level training process at the default 1e6 capacity. Plan about 3.5 GB per `train-low` job.
