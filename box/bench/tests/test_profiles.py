"""MacroHFT input profiles (MacroHFT/profiles/):

1. causality: every feature of every profile gives the same values at rows <= t whether or not later bars exist
   (catches centred windows, backward fills and any other look-ahead);
2. every built dataset under MacroHFT/data/<SYMBOL>_<TF>_<PROFILE>/ plugs into MacroHFT: the lists match the
   columns, all values are finite and within +-5, z-scaled features are ~N(0,1) on train, and the environment
   and the sub-agent / hyper-agent networks accept the state (shapes) and produce finite Q-values.
No training is run.
"""
import glob, os, sys
import numpy as np
import pandas as pd
import yaml

TM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MACRO = os.path.join(TM, "MacroHFT")
sys.path.insert(0, os.path.join(MACRO, "profiles")); sys.path.insert(0, TM)
from features import FEATURES, TRAIN_FITTED  # noqa: E402
import build  # noqa: E402


def test_causality(timeframe, start="2021-01-01", end="2021-06-30"):
    tf = yaml.safe_load(open(os.path.join(MACRO, "profiles", "timeframes.yaml")))[timeframe]
    freq = {"15m": "15min", "1h": "1h"}[tf["interval"]]
    k, _ = build.fill_gaps(build.load_klines("BTCUSDT", tf["interval"], pd.Timestamp(start), pd.Timestamp(end)), freq)
    n = len(k)
    train_mask = np.arange(n) < int(0.6 * n)
    rng = np.random.default_rng(0)
    cuts = sorted(set(rng.integers(int(0.62 * n), n - 1, size=6).tolist() + rng.integers(1300, int(0.58 * n), size=6).tolist()))
    specs = {}
    for p in glob.glob(os.path.join(MACRO, "profiles", "*.yaml")):
        if p.endswith("timeframes.yaml"):
            continue
        prof = yaml.safe_load(open(p))
        for s in prof["single"] + prof["trend"]:
            specs[(s["fn"], repr(build.resolve(s.get("args"), timeframe)))] = s
    checked = 0
    for spec in specs.values():
        full = build.compute(k, spec, timeframe, train_mask).to_numpy()
        for c in cuts:
            if spec["fn"] in TRAIN_FITTED and c < train_mask.sum():
                continue  # train-fitted: the fit itself uses the whole train split by design
            part = build.compute(k.iloc[:c + 1].reset_index(drop=True), spec, timeframe, train_mask[:c + 1]).to_numpy()
            a, b = full[:c + 1], part
            same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-9, atol=1e-12)
            assert same.all(), f"{spec['id']} ({timeframe}) changes at rows <= {c} when later bars are added"
            checked += 1
    print(f"causality {timeframe}: {len(specs)} distinct features x {len(cuts)} cut points ok ({checked} checks)")


def test_dataset(root):
    import torch
    from MacroHFT.model.net import subagent, hyperagent
    from MacroHFT.env.low_level_env import Testing_Env
    name = os.path.basename(root)
    single = np.load(os.path.join(root, "feature_list", "single_features.npy"), allow_pickle=True).tolist()
    trend = np.load(os.path.join(root, "feature_list", "trend_features.npy"), allow_pickle=True).tolist()
    scaling = yaml.safe_load(open(os.path.join(root, "feature_scaling.yaml")))
    assert single and trend, "both input groups must be non-empty"
    for split in ("train", "val", "test"):
        df = pd.read_feather(os.path.join(root, f"df_{split}.feather"))
        x = df[single + trend].to_numpy()
        assert np.isfinite(x).all() and np.isfinite(df.close).all(), f"{name}/{split}: non-finite"
        assert (np.abs(x[:, [scaling[f].get("std") is not None for f in single + trend]]) <= 5 + 1e-9).all()
        steps = pd.to_datetime(df.timestamp).diff().dropna().unique()
        assert len(steps) == 1, f"{name}/{split}: bars are not at a fixed interval"
        if split == "train":
            for f in single + trend:
                if scaling[f].get("std") is None:
                    continue
                if scaling[f]["transform"].endswith("robust"):  # median 0, IQR-based scale 1
                    q1, med, q3 = np.percentile(df[f], [25, 50, 75])
                    assert abs(med) < 0.1 and 0.8 < (q3 - q1) / 1.349 < 1.2, f"{name}: {f} not robust-scaled on train"
                else:
                    assert abs(df[f].mean()) < 0.1 and 0.8 < df[f].std() < 1.1, f"{name}: {f} not standardised on train"
    env = Testing_Env(df=df.iloc[:500].reset_index(drop=True), tech_indicator_list=single,
                      tech_indicator_list_trend=trend, max_holding_number=0.01, initial_action=0)
    s, s2, info = env.reset()
    sub = subagent(len(single), len(trend), 2, 64); hyp = hyperagent(len(single), len(trend), 2, 32)
    with torch.no_grad():
        for t in range(50):
            q = sub(torch.FloatTensor(s), torch.FloatTensor(s2), torch.tensor([info["previous_action"]]))
            w = hyp(torch.FloatTensor(s), torch.FloatTensor(s2), torch.zeros(1, 2), torch.tensor([info["previous_action"]]))
            assert torch.isfinite(q).all() and torch.isfinite(w).all()
            s, s2, r, done, info = env.step(int(q.argmax()))
    print(f"dataset {name}: {len(single)} single + {len(trend)} trend, finite, scaled, env + networks ok")


if __name__ == "__main__":
    os.chdir(MACRO)  # the env modules load ./data/feature_list at import
    for tf in ("15m", "1h"):
        test_causality(tf)
    roots = sorted(r for r in glob.glob(os.path.join(MACRO, "data", "*_*_*")) if os.path.exists(os.path.join(r, "feature_scaling.yaml")))
    for r in roots:
        test_dataset(r)
    print(f"profiles: {len(roots)} datasets ok")
