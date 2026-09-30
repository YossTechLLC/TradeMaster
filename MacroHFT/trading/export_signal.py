"""[TradeMaster] Export a trained high-level agent's full greedy signal (direction per bar) for tm_risk replay."""
import argparse
import dataclasses
import os
import re
import shutil
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MACRO = os.path.join(ROOT, "MacroHFT")
SCRATCH_EXP = "export_tmp"


def read_run_env(dataset):
    """--context_window / --memory_capacity (list of CLI tokens) from data/<dataset>/run.env, else []."""
    path = os.path.join(MACRO, "data", dataset, "run.env")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        m = re.search(r'^MACRO_HIGH_ARGS="([^"]*)"', f.read(), re.M)
    return m.group(1).split() if m else []


def parse_run(run_dir):
    """(result key '<ds>@<tag>', seed) from .../high_level/<key>/<exp>/seed_<n>."""
    parts = os.path.normpath(run_dir).split(os.sep)
    return parts[-3], int(parts[-1].split("_", 1)[1])


def trading_flags(run_dir):
    with open(os.path.join(run_dir, "trading_config.yaml")) as f:
        c = yaml.safe_load(f)["config"]
    return ["--trade_mode", c["mode"], "--venue", c["venue"]["name"], "--leverage", str(c["leverage"]),
            "--capital", str(c["capital"]), "--turnover_penalty_bps", str(c["turnover_penalty_bps"]),
            "--reward_scale", str(c["reward_scale"])]


def build_agent(run_dir, dataset):
    """DQN of the high-level module, built as high_level.py does; cwd must be MacroHFT/ (agents use relative paths)."""
    sys.path.insert(0, ROOT)
    os.chdir(MACRO)
    from MacroHFT.RL.agent import high_level as hl
    key, seed = parse_run(run_dir)
    if key.split("@")[0] != dataset:
        raise ValueError(f"run dir key {key!r} does not belong to dataset {dataset!r}")
    argv = (["--dataset", dataset, "--device", "cpu", "--seed", str(seed), "--exp", SCRATCH_EXP,
             "--subagent_path", os.path.join("result", "low_level", key, "best_model")]
            + read_run_env(dataset) + trading_flags(run_dir))
    agent = hl.DQN(hl.parser.parse_args(argv))
    if agent.result_key != key:
        raise ValueError(f"trading config gives result key {agent.result_key!r}, run dir has {key!r}")
    return agent


def export_split(agent, run_dir, split):
    """Greedy act_test over whole/<split>.feather with a fixed-sizing env; writes <run_dir>/<split>/signal_<split>.npz."""
    from MacroHFT.trading.envs import make_env
    cfg = dataclasses.replace(agent.trading, sizing="fixed").validate()
    agent.hyperagent.load_state_dict(__import__("torch").load(os.path.join(run_dir, "best_model.pkl"), map_location="cpu"))
    agent.hyperagent.eval()
    df = agent.read_split(os.path.join(MACRO, "data", agent.dataset, "whole", f"{split}.feather"))
    env = make_env("test", "high", df, agent.tech_indicator_list, agent.tech_indicator_list_trend, cfg,
                   clf_list=agent.clf_list, transcation_cost=agent.transcation_cost,
                   back_time_length=agent.back_time_length, max_holding_number=agent.max_holding_number,
                   initial_action=agent.eval_initial_action)
    s, s2, s3, info = env.reset()
    actions, done = [], False
    while not done:
        a = int(agent.act_test(s, s2, s3, info))
        s, s2, s3, _, done, info = env.step(a)
        actions.append(a)
    actions = np.array(actions)
    # step i holds the position over df row stack+i (decided at the close of the row before)
    rows = env.stack_length + np.arange(len(actions))
    ts = df["timestamp"].to_numpy()[rows] if "timestamp" in df.columns else rows
    direction = np.array(cfg.directions)[actions].astype(np.int8)
    out = os.path.join(run_dir, split)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"signal_{split}.npz"), timestamp=ts, direction=direction, df_row=rows, action=actions)
    old = os.path.join(out, "action.npy")
    mism = None
    if os.path.isfile(old):
        ref = np.load(old).reshape(-1)
        n = min(len(ref), len(actions))
        mism = int((ref[:n] != actions[:n]).sum())      # over the overlap; old logs end early at bankruptcy
        note = f" (old log truncated at {len(ref)})" if len(ref) < len(actions) else ""
        print(f"{run_dir} {split}: {len(actions)} rows, action.npy {len(ref)} rows{note}, mismatches {mism}")
    else:
        print(f"{run_dir} {split}: {len(actions)} rows (no action.npy to compare)")
    return mism


def main(argv=None):
    ap = argparse.ArgumentParser(description="export the full greedy signal of a high-level run")
    ap.add_argument("--run_dir", required=True, help="result/high_level/<ds>@<tag>/<exp>/seed_<n>")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args(argv)
    run_dir = os.path.abspath(a.run_dir)
    scratch = os.path.join(os.path.dirname(os.path.dirname(run_dir)), SCRATCH_EXP)
    if os.path.exists(scratch):
        raise SystemExit(f"scratch dir {scratch} exists; remove it first")
    try:
        agent = build_agent(run_dir, a.dataset)
        agent.writer.close()
        total = 0
        for split in a.splits:
            m = export_split(agent, run_dir, split)
            total += m or 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("parity mismatches:", total)
    return total


if __name__ == "__main__":
    main()
