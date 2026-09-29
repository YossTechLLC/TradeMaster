# MacroHFT in TradeMaster

**Paper:** *MacroHFT: Memory Augmented Context-aware Reinforcement Learning On High Frequency Trading* (KDD 2024), [PDF](https://personal.ntu.edu.sg/boan/papers/KDD24_MacroHFT.pdf) · [arXiv 2406.14537](https://arxiv.org/abs/2406.14537) (local copy: `articles/MacroHFT.pdf`)
**Upstream:** https://github.com/ZONG0004/MacroHFT @ `31e5ef41f93b2aea6e63e6ae2267675c997a8e54` (2025-01-09), vendored without history.
The upstream `readme.md` is kept. This file explains how the project is wired into TradeMaster.

## What it does

MacroHFT trades crypto at minute level (ETHUSDT in the demo) in two phases.

| Phase (paper) | Idea | Code |
| --- | --- | --- |
| Market decomposition | Cut the data into chunks of 4320 minutes (3 days). Label each chunk by **trend** (slope of a low-pass-filtered linear fit) and by **volatility**, split into quantile terciles. Validation and test sets reuse the training thresholds. | `preprocess/decomposition.py` |
| I. Sub-agents | One dueling DDQN per market type: bear, flat, bull, stable, medium, volatile. Each has a **conditional adapter**: context features and the current position shift and scale the hidden state through LayerNorm modulation. A DP "optimal action" demonstration adds a KL term to the loss. | `RL/agent/low_level.py`, `model/net.py` (`subagent`, `modulate`), `tools/demonstration.py`, `env/low_level_env.py` |
| II. Hyper-agent | Mixes the six sub-agents' Q-values with softmax weights conditioned on market context. An **episodic memory** (kernel-weighted lookup of similar past states) stabilises it in extreme markets. | `RL/agent/high_level.py`, `model/net.py` (`hyperagent`, `calculate_q`), `RL/util/memory.py`, `env/high_level_env.py` |

Features: `data/feature_list/single_features.npy` (36 single-state features) and `trend_features.npy` (9 context features). Both ship with the repo and are loaded when the env modules are imported.

## Layout

```
MacroHFT/
├── run.sh                        TradeMaster launcher (single GPU, .venv-hft, foreground, logs/)
├── data/feature_list/            feature lists (shipped)
├── data/ETHUSDT/                 df_{train,val,test}.feather from Google Drive + decomposition output (git-ignored)
├── result/low_level/ETHUSDT/best_model/{slope,vol}/{1,2,3}/best_model.pkl   pretrained sub-agents (shipped, tracked)
├── result/high_level/            hyper-agent runs (git-ignored)
└── scripts/                      upstream multi-GPU scripts (reference only)
```

The code imports itself as the `MacroHFT` package (`from MacroHFT.model.net import *`), with the TradeMaster root on `sys.path`. **Keep the folder named `MacroHFT`.**

## Running it

Environment: the shared `../.venv-hft` (Python 3.10, torch 2.0.1+cu118). See the root README.

1. Download the `ETHUSDT` folder from the Google Drive link in `readme.md` into `MacroHFT/data/`, so that `data/ETHUSDT/df_train.feather`, `df_val.feather` and `df_test.feather` exist.
2. Run the stages:

```bash
./run.sh help
./run.sh decompose                    # Step 1: chunk and label -> data/ETHUSDT/{train,val,test,whole}/
./run.sh train-low                    # Step 2 (optional): retrain all six sub-agents one after another
./run.sh train-low slope label_1 1    #   ...or one sub-agent: <clf> <label> <alpha>
./run.sh train-high                   # Step 3: hyper-agent over the six sub-agents
```

The six pretrained sub-agents ship in `result/low_level/…/best_model/`, so after `decompose` you can go straight to `train-high`. `DEVICE=cuda:0` is the default. Anything after `--` is passed to Python.

## Adaptations vs upstream

- `run.sh` runs every job on one device in the foreground. Upstream used `cuda:0-3` with `nohup … &` and expected `logs/…` to already exist.
- **Two bug fixes** in `RL/agent/low_level.py` and `RL/agent/high_level.py`, marked `# [TradeMaster]`:
  1. The end-of-training `torch.save(best_model.state_dict(), …)` called `.state_dict()` on something that was already a state dict, so it raised `AttributeError` after all epochs finished. The best weights are now also deep-copied when selected; before, they were a live reference to the final weights.
  2. `low_level.py` saved its best model to `result/low_level/<ds>/<clf>/label_<n>/best_model.pkl`, a directory that is never created, and not where `high_level.py` loads sub-agents from. It now saves to `result/low_level/<ds>/best_model/<clf>/<n>/best_model.pkl`, so Step 2 feeds Step 3 directly. Retraining therefore **overwrites the shipped checkpoints**, which you can restore with `git checkout MacroHFT/result`.
- `decomposition.py` and `high_level.py` hard-code ETHUSDT. Using another pair means editing those paths.
