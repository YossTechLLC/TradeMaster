"""Golden-output harness: proves that performance changes leave results unchanged.

It runs short, seeded, single-threaded CPU jobs through the real entry points, then
fingerprints everything they write: saved .npy results, model state_dicts, and the
stdout lines that carry results (the 'portfit margine' prints). Two fingerprints are
compared bit for bit.

usage: golden.py prepare                 # build the small MacroHFT BTCGOLD / EarnHFT BTCGOLD datasets (once)
       golden.py run <name> [macro|earn|all]   # run the jobs and write golden/<name>.json
       golden.py compare <a> <b>         # diff two fingerprints; exit 1 if they differ

The prepared inputs are frozen on purpose: 'prepare' never overwrites them, so later
changes to the preprocessing code cannot silently change the golden inputs.
"""
import glob, hashlib, json, os, pickle, re, shutil, subprocess, sys, time
import numpy as np
import pandas as pd

TM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MACRO = os.path.join(TM, "MacroHFT")
EARN = os.path.join(TM, "EarnHFT", "EarnHFT_Algorithm")
PY = os.environ.get("PY", os.path.join(TM, ".venv-hft", "bin", "python"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
DS = "BTCGOLD"
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           TM_THREADS="1", PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="")


# ----------------------------------------------------------------------------- prepare
def prepare_macro():
    root = os.path.join(MACRO, "data", DS)
    if os.path.exists(root):
        print(f"{root} exists, keeping it"); return
    src = os.path.join(MACRO, "data", "BTCUSDT")
    parts = {k: pd.read_feather(os.path.join(src, f"df_{k}.feather")) for k in ("train", "val", "test")}
    # sub-agent chunks: 3 train / 2 val / 1 test chunks of 1500 bars, every label sees every chunk
    for split, n in (("train", 3), ("val", 2), ("test", 1)):
        d = os.path.join(root, split); os.makedirs(d)
        for i in range(n):
            parts[split][i * 1500:(i + 1) * 1500].reset_index(drop=True).to_feather(os.path.join(d, f"df_{i}.feather"))
        idx = [[], list(range(n)), list(range(n)), list(range(n)), []]
        for clf in ("slope", "vol"):
            with open(os.path.join(d, f"{clf}_labels.pkl"), "wb") as f:
                pickle.dump(idx, f)
    # hyper-agent 'whole' files: slope_360 / vol_360 context computed with the upstream code
    sys.path.insert(0, os.path.join(MACRO, "preprocess"))
    from decomposition import label_whole
    d = os.path.join(root, "whole"); os.makedirs(d)
    for split, n in (("train", 6000), ("val", 1500), ("test", 1500)):
        df = parts[split][: n + 361].reset_index(drop=True)
        df = label_whole(df).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
        df.to_feather(os.path.join(d, f"{split}.feather"))
        print(f"{DS} whole/{split}: {len(df)} bars")


EARN_CHUNK, EARN_SIGHT = 3000, 600  # golden low-level chunks: 3000 steps + 600 look-ahead rows


def prepare_earn():
    root = os.path.join(EARN, "data", DS)
    if os.path.exists(root):
        print(f"{root} exists, keeping it"); return
    src = os.path.join(EARN, "data", "BTCUSDT", "df.feather")
    if not os.path.exists(src):
        print(f"skip EarnHFT: {src} missing (build_data.py earn ...)"); return
    df = pd.read_feather(src)
    tr = os.path.join(root, "train"); os.makedirs(tr)
    train = df[:8 * EARN_CHUNK + 800].reset_index(drop=True)  # like split_data.py: 8 full chunks (the PES selector needs >= 5) + a short tail
    for i, s in enumerate(range(0, len(train), EARN_CHUNK)):
        train[s:s + EARN_CHUNK + EARN_SIGHT].reset_index(drop=True).to_feather(os.path.join(tr, f"df_{i}.feather"))
    train.to_feather(os.path.join(root, "train.feather"))
    rest = df[len(train):].reset_index(drop=True)
    valid, test = rest[:3000].reset_index(drop=True), rest[3000:6000].reset_index(drop=True)
    valid.to_feather(os.path.join(root, "valid.feather")); test.to_feather(os.path.join(root, "test.feather"))
    # hand-made market-dynamics segments (the DTW labeller is far too slow for a golden run)
    for k in range(5):
        d = os.path.join(root, "valid", f"label_{k}"); os.makedirs(d)
        for j in range(2):
            s = (2 * k + j) * 300
            valid[s:s + 300].reset_index(drop=True).to_feather(os.path.join(d, f"df_{j}.feather"))
    print(f"{DS} EarnHFT: {len(train)} train rows in chunks of {EARN_CHUNK}+{EARN_SIGHT}, 3000-row valid/test")


# ----------------------------------------------------------------------------- run
def sh(cmd, cwd, log):
    t = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log, "w") as f:
        f.write(p.stdout)
    print(f"  {os.path.basename(log)}: exit {p.returncode}, {time.time() - t:.1f} s")
    if p.returncode:
        print(p.stdout[-3000:]); raise SystemExit(f"job failed: {' '.join(cmd)}")
    return p.stdout


# Reported metrics (return rate, required money, final balance) are compared with a relative tolerance
# of 1e-12: np.cumsum and the upstream per-prefix np.sum can differ in the last ulp (rounding to N digits
# is not enough - a 1-ulp change can straddle a rounding boundary). Actions, rewards fed to training and
# model weights are compared bit for bit.
FLOAT = re.compile(r"-?\d+\.\d+(?:e[-+]?\d+)?|-?\d+e[-+]?\d+|nan|inf")
REPORTED = ("final_balance", "require_money", "return_rate")


def result_lines(stdout):
    keep = re.compile(r"portfit margine|best model updated|we are training with|training with df|validating on df")
    return [l.strip() for l in stdout.splitlines() if keep.search(l)]


def close(a, b, rel=1e-12):
    import math
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str):  # stdout line: same text, numbers within tolerance
        na, nb = FLOAT.findall(a), FLOAT.findall(b)
        return FLOAT.sub("#", a) == FLOAT.sub("#", b) and close([float(x) for x in na], [float(y) for y in nb], rel)
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
    return a == b


def digest_file(path):
    if path.endswith(".npy"):
        a = np.load(path, allow_pickle=True)
        if any(k in os.path.basename(path) for k in REPORTED) and a.dtype.kind == "f" and a.size <= 100:
            return [float(x) for x in a.ravel()]  # compared with close()
        return hashlib.sha256(repr(a.tolist()).encode()).hexdigest()[:16]
    if path.endswith((".pkl", ".pth")):
        import torch
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict):
            h = hashlib.sha256()
            for k in sorted(sd):
                v = sd[k]
                h.update(k.encode()); h.update(v.cpu().numpy().tobytes() if hasattr(v, "cpu") else repr(v).encode())
            return h.hexdigest()[:16]
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def fingerprint_tree(root):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.isfile(p) and p.endswith((".npy", ".pkl", ".pth")) and "/log/" not in p:
            out[os.path.relpath(p, root)] = digest_file(p)
    return out


def help_has(script, cwd, flag):
    p = subprocess.run([PY, script, "--help"], cwd=cwd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return flag in p.stdout


def run_macro(name, logs):
    fp = {}
    for d in ("result/low_level/" + DS, "result/high_level/" + DS):
        shutil.rmtree(os.path.join(MACRO, d), ignore_errors=True)
    out = sh([PY, "-u", "RL/agent/low_level.py", "--dataset", DS, "--device", "cpu", "--clf", "slope",
              "--label", "label_1", "--alpha", "1", "--epoch_number", "2"], MACRO, os.path.join(logs, "macro_low.log"))
    fp["macro_low_stdout"] = result_lines(out)
    fp["macro_low_files"] = fingerprint_tree(os.path.join(MACRO, "result/low_level", DS))
    high = [PY, "-u", "RL/agent/high_level.py", "--dataset", DS, "--device", "cpu", "--epoch_number", "2"]
    if help_has("RL/agent/high_level.py", MACRO, "--subagent_path"):
        high += ["--subagent_path", GOLD_SUBAGENTS]
    t = time.time()
    out = sh(high, MACRO, os.path.join(logs, "macro_high.log"))
    fp["macro_high_seconds"] = round(time.time() - t, 1)
    fp["macro_high_stdout"] = result_lines(out)
    fp["macro_high_files"] = fingerprint_tree(os.path.join(MACRO, "result/high_level", DS))
    return fp


# the hyper-agent needs six sub-agents; the golden run always uses the shipped upstream ones
GOLD_SUBAGENTS = os.environ.get("GOLD_SUBAGENTS", "./result/low_level/ETHUSDT/best_model")


def run_earn(name, logs):
    if not os.path.exists(os.path.join(EARN, "data", DS)):
        print("  EarnHFT golden data missing, skipped"); return {}
    from golden_earn import run as earn_run  # EarnHFT jobs live next to this file
    return earn_run(sys.modules[__name__], logs)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "prepare":
        prepare_macro(); prepare_earn()
    elif cmd == "run":
        name, which = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "all")
        logs = os.path.join(OUT, name + "_logs"); os.makedirs(logs, exist_ok=True)
        path = os.path.join(OUT, name + ".json")
        fp = json.load(open(path)) if os.path.exists(path) else {}
        if which in ("macro", "all"):
            fp.update(run_macro(name, logs))
        if which in ("earn", "all"):
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            fp.update(run_earn(name, logs))
        json.dump(fp, open(path, "w"), indent=1, sort_keys=True)
        print(f"wrote {path}")
    elif cmd == "compare":
        a, b = (json.load(open(os.path.join(OUT, n + ".json"))) for n in sys.argv[2:4])
        bad = 0
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                print(f"  {k}: only in one run, skipped"); continue
            if k.endswith("_seconds"):
                print(f"  {k}: {a.get(k)} -> {b.get(k)}"); continue
            va, vb = a[k], b[k]
            if isinstance(va, dict) and isinstance(vb, dict):
                diffs = [f for f in sorted(set(va) | set(vb)) if not close(va.get(f), vb.get(f))]
            else:
                diffs = [] if close(va, vb) else ["value"]
            if diffs:
                bad += 1
                print(f"DIFF {k}")
                for f in diffs[:20]:
                    print(f"   {f}: {(va.get(f), vb.get(f)) if isinstance(va, dict) else (va, vb)!r:.400}")
        print("IDENTICAL" if not bad else f"{bad} keys differ")
        sys.exit(1 if bad else 0)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
