"""Differential test: vectorised Q-teacher tables == the upstream per-row loops, bit for bit."""
import glob, os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM); sys.path.insert(0, HERE)


def check(name, new_fn, old_fn, df, n_old, **kw):
    part = df.iloc[:n_old].reset_index(drop=True)
    t = time.time(); old = old_fn(part, **kw); t_old = time.time() - t
    t = time.time(); new = new_fn(part, **kw); t_new = time.time() - t
    assert old.shape == new.shape and np.array_equal(old, new), (name, np.abs(old - new).max())
    print(f"{name}: {n_old} rows bit-identical; upstream {t_old:.2f} s, vectorised {t_new:.3f} s")


def macro():
    from MacroHFT.tools.demonstration import make_q_table_reward as new
    from _upstream_macro_demonstration import make_q_table_reward as old
    df = pd.read_feather(os.path.join(TM, "MacroHFT/data/BTCGOLD/whole/train.feather"))
    for kw in (dict(num_action=2, max_holding=0.01, commission_fee=0.001, reward_scale=1, gamma=0.99, max_punish=1e12),
               dict(num_action=2, max_holding=0.2, commission_fee=0.0002, reward_scale=1, gamma=0.99, max_punish=1e12),
               dict(num_action=3, max_holding=0.2)):
        check("MacroHFT q_table " + str(kw), new, old, df, 3000, **kw)


def earn():
    p = os.path.join(TM, "EarnHFT/EarnHFT_Algorithm")
    if not glob.glob(os.path.join(p, "data/BTCGOLD/train/df_0.feather")):
        print("EarnHFT: golden data missing, skipped"); return
    sys.path.insert(0, p)
    from tool.demonstration import make_q_table_reward as new
    from _upstream_earn_demonstration import make_q_table_reward as old
    df = pd.read_feather(os.path.join(p, "data/BTCGOLD/train/df_0.feather"))
    for kw in (dict(num_action=5, max_holding=0.01, commission_fee=0, reward_scale=1, gamma=1, max_punish=1e12),
               dict(num_action=5, max_holding=0.01, commission_fee=0.00015, reward_scale=1, gamma=0.99, max_punish=1e12),
               dict(num_action=5, max_holding=5.0, commission_fee=0.00015, reward_scale=1, gamma=1, max_punish=1e12)):
        check("EarnHFT q_table " + str(kw), new, old, df, 4000, **kw)


if __name__ == "__main__":
    macro(); earn()
