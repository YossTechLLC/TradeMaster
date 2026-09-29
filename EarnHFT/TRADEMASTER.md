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
├── run.sh                 TradeMaster launcher (single GPU, .venv-hft, foreground, logs)
├── data_preprocess/       Tardis download -> clean -> features -> merge -> IC feature selection
└── EarnHFT_Algorithm/     RL code; every script expects to run from this directory
    ├── data/<PAIR>/       df.feather, then train/ valid/ test splits (generated, git-ignored)
    ├── data/feature/      second_feature.npy, minitue_feature.npy (generated, loaded at import time)
    ├── result_risk/<PAIR>/  low-level agents, potential_model/ pool, high_level/ router (git-ignored)
    └── script/<PAIR>/     upstream multi-GPU scripts (reference only)
```

## Running it

Environment: the shared `../.venv-hft` (Python 3.10, torch 2.0.1+cu118). See the root README.

```bash
./run.sh help
./run.sh download   ETHUSDT 2022-05-01 2022-06-15   # needs a Tardis API key + pip install tardis-dev nest_asyncio
./run.sh preprocess ETHUSDT 2022-05-01 2022-06-15   # also copies df.feather + feature lists into EarnHFT_Algorithm/data/
./run.sh split      ETHUSDT
./run.sh label      ETHUSDT
./run.sh train-low  ETHUSDT            # betas -10 -90 30 100, run one after another (or: train-low ETHUSDT 30)
./run.sh valid-low  ETHUSDT
./run.sh pick       ETHUSDT
./run.sh train-high ETHUSDT
./run.sh test-high  ETHUSDT            # or a single epoch: test-high ETHUSDT result_risk/ETHUSDT/high_level/seed_12345/epoch_58
```

Anything after `--` is passed to the Python script, for example `./run.sh train-low ETHUSDT 30 -- --epoch_number 5`.

### Data requirements

- **Tardis.dev** (`book_snapshot_5` + `trades`) is the only supported source, and full date ranges need a paid API key. Put the key in `data_preprocess/download_code/download.py` (`api_key=`). `tardis-dev` is left out of `requirements-hft.txt` on purpose because it is only needed for downloading.
- Nothing ships with the repo. The RL scripts fail at import time with `FileNotFoundError: data/feature/second_feature.npy` until `preprocess` has run.
- If you bring your own LOB data, match the columns described in `data_preprocess/README.md` and write `EarnHFT_Algorithm/data/<PAIR>/df.feather` and the two feature `.npy` lists yourself.

## Adaptations vs upstream

- **No source changes.** All wiring is in `run.sh`.
- Upstream scripts spread jobs across GPUs 0–3 with `nohup … &`. `run.sh` runs them one after another on `CUDA_VISIBLE_DEVICES` (default 0).
- Upstream `script/<PAIR>/high_level/pick.sh` calls `analysis/pick_agent_position.py`, which doesn't exist. The real path is `analysis/pick_agent/pick_agent_position.py`, and `run.sh` uses it.
- Per-pair `--max_holding_number` and `--transcation_cost` follow the upstream scripts: ETHUSDT 0.1, GALAUSDT 4000, BTCUSDT 0.01, BTCTUSD 0.01 with zero fee. Upstream `valid_multi.sh` leaves these at their defaults (0.01). `run.sh` passes the pair's values to every stage so train, valid and test use the same position size. You can override them with `MAX_HOLD=… TC=…`.
- `data/feature/*.npy` is one shared pair of files (an upstream design choice). The last pair you ran `preprocess` on is what every pair will use.
- The router's agent-pool paths are hard-coded per pair in `RL/agent/high_level/dqn_position.py` for the four pairs above. A new pair needs its own branch there.
