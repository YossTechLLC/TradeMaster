"""EarnHFT golden jobs (imported by golden.py): every pipeline stage on the tiny BTCGOLD set.

train-low (2 betas x 8 samples = 2 checkpoints each) -> valid-low (every epoch x 5 initial positions)
-> pick -> train-high (router, 2 passes) -> test-high (every router epoch on valid + test).
"""
import os, shutil, time

ARGS = ["--max_holding_number", "0.01", "--transcation_cost", "0.00015"]


def run(g, logs):
    E, DS = g.EARN, g.DS
    shutil.rmtree(os.path.join(E, "result_risk", DS), ignore_errors=True)
    fp, t0 = {}, time.time()
    out = ""
    for beta in ("-10", "30"):
        t = time.time()
        out += g.sh([g.PY, "-u", "RL/agent/low_level/ddqn_pes_risk_aware.py", "--beta", beta, "--train_data_path",
                     f"data/{DS}/train", "--dataset_name", DS, "--num_sample", "8", "--chunk_length", str(g.EARN_CHUNK)] + ARGS,
                    E, os.path.join(logs, f"earn_low_{beta}.log"))
        fp[f"earn_low_{beta}_seconds"] = round(time.time() - t, 1)
    fp["earn_low_stdout"] = g.result_lines(out)
    t, out = time.time(), ""
    for beta in ("-10.0", "30.0"):
        root = f"result_risk/{DS}/beta_{beta}_risk_bond_0.1/seed_12345"
        for ep in sorted(e for e in os.listdir(os.path.join(E, root)) if e.startswith("epoch_")):
            for a in range(5):
                out += g.sh([g.PY, "-u", "RL/agent/low_level/test_ddqn.py", "--test_path", f"{root}/{ep}", "--initial_action",
                             str(a), "--test_df_path", f"data/{DS}/valid"] + ARGS, E, os.path.join(logs, "earn_valid.log"))
    fp["earn_valid_seconds"] = round(time.time() - t, 1)
    fp["earn_valid_stdout"] = g.result_lines(out)
    out = g.sh([g.PY, "-u", "analysis/pick_agent/pick_agent_position.py", "--root_path", f"result_risk/{DS}",
                "--save_path", f"result_risk/{DS}/potential_model"], E, os.path.join(logs, "earn_pick.log"))
    fp["earn_pick_stdout"] = [l for l in out.splitlines() if "epoch" in l]
    t = time.time()
    out = g.sh([g.PY, "-u", "RL/agent/high_level/dqn_position.py", "--train_data_path", f"data/{DS}/train.feather",
                "--dataset_name", DS, "--num_sample", "2"] + ARGS, E, os.path.join(logs, "earn_router.log"))
    fp["earn_router_seconds"] = round(time.time() - t, 1)
    fp["earn_router_stdout"] = g.result_lines(out)
    t, out = time.time(), ""
    hl = f"result_risk/{DS}/high_level/seed_12345"
    for ep in sorted(e for e in os.listdir(os.path.join(E, hl)) if e.startswith("epoch_")):
        out += g.sh([g.PY, "-u", "RL/agent/high_level/test_dqn_position.py", "--test_path", f"{hl}/{ep}", "--dataset_name", DS,
                     "--valid_data_path", f"data/{DS}/valid.feather", "--test_data_path", f"data/{DS}/test.feather"] + ARGS,
                    E, os.path.join(logs, "earn_test_high.log"))
    fp["earn_test_high_seconds"] = round(time.time() - t, 1)
    fp["earn_test_high_stdout"] = g.result_lines(out)
    fp["earn_files"] = g.fingerprint_tree(os.path.join(E, "result_risk", DS))
    fp["earn_total_seconds"] = round(time.time() - t0, 1)
    return fp
