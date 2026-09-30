import pandas as pd
import numpy as np
import os
import re
import argparse

# [TradeMaster] upstream picked the router epoch by its *test* return ("valid_path" pointed at test/),
# from a hard-coded result_risk/BTCTUSD/ppo path and the first 84 epochs. It now selects on the valid set
# and only reports the chosen epoch's test return. Used by `run.sh pick-high <PAIR>`.
parser = argparse.ArgumentParser()
parser.add_argument("--high_level_path", type=str, default="result_risk/BTCTUSD/high_level/seed_12345",
                    help="router run directory with epoch_<n>/{valid,test}/ from test_dqn_position.py")


def sort_list(lst: list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    lst.sort(key=alphanum_key)


def return_rate(path):
    return np.load(os.path.join(path, "final_balance.npy")) / (np.load(os.path.join(path, "require_money.npy")) + 1e-12)


if __name__ == "__main__":
    args = parser.parse_args()
    high_level_path = args.high_level_path
    epoch_list = [e for e in os.listdir(high_level_path)
                  if e.startswith("epoch_") and os.path.exists(os.path.join(high_level_path, e, "valid", "final_balance.npy"))]
    if not epoch_list:
        raise SystemExit("no tested router epochs under {} (run test-high first)".format(high_level_path))
    sort_list(epoch_list)
    valid_result = [float(return_rate(os.path.join(high_level_path, e, "valid"))) for e in epoch_list]
    best_index = valid_result.index(max(valid_result))
    best_epoch = epoch_list[best_index]
    for e, r in zip(epoch_list, valid_result):
        print("{:>10}  valid return rate {: .6f}".format(e, r))
    print("best epoch on valid:", best_epoch)
    print("its valid return rate:", valid_result[best_index])
    print("its test return rate:", float(return_rate(os.path.join(high_level_path, best_epoch, "test"))))
