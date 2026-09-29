# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environments (two, not interchangeable)

| venv | Python | Stack | Used by |
|---|---|---|---|
| `.venv` | 3.9 | torch 1.12.1+cu113, ray[rllib] 1.13, tensorflow 2.11, mmcv 1.7.1, gym 0.21 | core `trademaster/`, `tools/`, `configs/` |
| `.venv-hft` | 3.10 | torch 2.0.1+cu118, numpy 1.24, pandas 1.5.3, gym 0.21, mmcv 1.7.1 | `EarnHFT/`, `MacroHFT/`, `Market-GAN/` |

```bash
# core
uv venv --python 3.9 --seed .venv
.venv/bin/python -m pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40"
.venv/bin/python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
.venv/bin/python -m pip install -r requirements.txt -c constraints.txt
# extensions
uv venv --python 3.10 --seed .venv-hft
.venv-hft/bin/python -m pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40"
.venv-hft/bin/python -m pip install -r requirements-hft.txt
```

The pins are load-bearing. Don't upgrade them casually:
- `constraints.txt`: `numpy<1.24` because ray 1.13 uses `np.bool`, and `protobuf<3.20` for tf/ray.
- `yapf<=0.40.1` in both environments: mmcv 1.7.1 calls `FormatCode(verify=...)`, and newer yapf crashes on `cfg.dump`.
- `pydantic==1.10.26`: 1.10.2 breaks with modern `typing_extensions`.
- The old `setuptools`/`wheel`/`pip` versions are required to build the `gym==0.21.0` sdist.

`tools/earnmore` and `tools/finagent` (with the `pm/` and `finagent/` packages) need `mmengine`, which neither venv includes. They have their own `requirements.txt`.

## Running / verifying

Run everything from the repo root. Config data paths such as `data/algorithmic_trading/FX/train.csv` are relative to the working directory.

```bash
.venv/bin/python tools/algorithmic_trading/train.py                       # canonical install check (DeepScalper on FX, 20 epochs)
.venv/bin/python tools/portfolio_management/train.py                      # exercises ray/RLlib (PPO)
.venv/bin/python tools/<task>/train.py --config configs/<task>/<file>.py --task_name test   # train | test | dynamics_test
```

Outputs go to `work_dir/<config name>/`, which is git-ignored.

**Tests:** `unit_testing/` is stale. It references configs that don't exist (e.g. `dqn_btc.py`), one file has a syntax error, and pytest isn't installed. Don't treat it as a test suite. Verify changes by running the relevant `tools/*/train.py`. There is no lint configuration.

## Core architecture (`trademaster/`)

This is an mmcv config-driven registry framework. Every component type (`agents`, `datasets`, `environments`, `nets`, `losses`, `optimizers`, `trainers`, `transition`, `preprocessor`, `imputation`, `collector`) follows the same layout:
- `builder.py` defines an `mmcv.utils.Registry` and a `build_x(cfg, default_args)` function.
- `custom.py` defines the base class.
- Implementations live in per-task subfolders (`algorithmic_trading`, `portfolio_management`, `order_execution`, `high_frequency_trading`) and are decorated with `@X.register_module()`.
- **Registration happens through imports in each package's `__init__.py`.** A new class must be imported there, or `build_*` won't find it.

Configs:
- `configs/<task>/<task>_<dataset>_<net>_<agent>_<optimizer>_<loss>.py` inherits `_base_` fragments from `configs/_base_/{datasets,environments,agents,trainers,losses,optimizers,nets}/`, then overrides fields.
- `type=` in each dict selects the registered class.
- `work_dir` is derived from the file name.

A `tools/<task>/train.py` script runs this sequence:
1. `Config.fromfile`, then `replace_cfg_vals`.
2. `build_dataset`, then three `build_environment` calls (train, valid, test) with `default_args=dict(dataset=..., task=...)`.
3. It reads `state_dim`/`action_dim` from the env and injects them into `cfg.act`/`cfg.cri` before `build_net`.
4. Optimizer, loss, transition, then `build_agent` (with nets and optimizers as `default_args`), then `build_trainer`.
5. It calls `trainer.train_and_valid()` or `trainer.test()`, and dumps the resolved config to `work_dir`.

Keep new tasks in this shape.

Other components:
- `trademaster/environments/*` are `gym.Env` subclasses (gym 0.21 API).
- Portfolio-management trainers wrap environments for ray RLlib (`ray.rllib.agents.*`).
- `tools/market_dynamics_labeling/` labels bull/bear/sideways regimes. Its labelled CSVs feed the `dynamics_test` evaluation mode.
- `pm/` and `finagent/` are separate mmengine-registry packages (EarnMore and FinAgent). They don't use the mmcv registries above.

## Vendored extensions: EarnHFT/, MacroHFT/, Market-GAN/

These are upstream repositories copied in without history; the upstream commit is recorded in each `TRADEMASTER.md`.

**Read `<project>/TRADEMASTER.md` first.** It maps paper components to files, lists pipeline stages and data prerequisites, and records every deviation from upstream.

Rules for working in them:
- Each has a `run.sh` launcher (`./run.sh help`). It sets the correct working directory, uses `.venv-hft`, runs on one GPU in the foreground, and writes logs.
  - The launcher `cd`s for you. Upstream code assumes cwd-relative paths (`./data`, `./result`, `data/feature/*.npy`).
  - Upstream `script*/` and `*.sh` files are kept as reference only. They hard-code `cuda:1-3` and `nohup … &`.
- Keep upstream code unmodified where possible. Mark any necessary fix with `# [TradeMaster]` and list it in that project's `TRADEMASTER.md`.
- `MacroHFT` imports itself as a package (`from MacroHFT.model.net import *`, with the repo root on `sys.path`). The folder name must stay `MacroHFT`.
- Env modules load feature lists at import time. EarnHFT's `data/feature/*.npy` doesn't exist until its `data_preprocess` pipeline has run, so import errors there usually mean missing data, not a code bug.
- Data sources:
  - **EarnHFT:** Tardis.dev 5-level order book + trades, with a paid key.
  - **MacroHFT:** Google Drive download (ETHUSDT). Six pretrained sub-agents are tracked under `result/low_level/ETHUSDT/best_model/`.
  - **Market-GAN:** ships `data/DJI/DJI_data.csv`. Run `./run.sh mdm` before training.
- The per-sub-agent networks are tiny MLPs, so training is CPU/simulator-bound. The CPU was measured faster than the GPU for the per-step action selection that dominates training.

## Remote compute

`box/init_box.sh` sets up this repo on a borrowed host (ssh alias `chad-box`, into `/home/chad/TradeMaster` only). It has stages `check`, `sync`, `venv-hft`, `venv-core`, `smoke`, `bench` and `status`. Host rules:
- Keep total RAM at or below 48 GB, and wrap every heavy command in `systemd-run --user --scope -p MemoryMax=…`.
- Keep at least 20 GiB of disk free.
- Never touch the owner's files or processes (`SIMONS_v3`, other GPU jobs).
- Copying files to the host needs the user's explicit approval.
