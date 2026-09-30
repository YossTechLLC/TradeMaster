import numpy as np
import pandas as pd


# [TradeMaster] vectorised rewrite of the upstream per-row loop (df.iloc per row, ~0.5 ms/row).
# Rewards are computed for all rows at once with the same float operations in the same order,
# then the backward recursion runs over plain numpy rows, so the table is bit-identical to upstream
# (box/bench/tests/test_q_tables.py). Upstream loop: box/bench/tests/_upstream_macro_demonstration.py
def make_q_table_reward(df: pd.DataFrame,
                        num_action,
                        max_holding,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.001,
                        max_punish=1e12):
    q_table = np.zeros((len(df), num_action, num_action))
    if len(df) < 2:
        return q_table
    close = df["close"].to_numpy(dtype=np.float64)
    current_close, future_close = close[:-1], close[1:]  # row i: price at i, price at i + 1

    scale_factor = num_action - 1
    reward = np.empty((len(df) - 1, num_action, num_action))
    for previous_action in range(num_action):
        for current_action in range(num_action):
            previous_position = previous_action / (scale_factor) * max_holding
            current_position = current_action / (scale_factor) * max_holding
            current_value = current_close * previous_position
            future_value = future_close * current_position
            if current_action > previous_action:
                position_change = (current_action - previous_action) / scale_factor * max_holding
                buy_money = position_change * current_close * (1 + commission_fee)
                r = future_value - (current_value + buy_money)
            else:
                position_change = (previous_action - current_action) / scale_factor * max_holding
                sell_money = position_change * current_close * (1 - commission_fee)
                r = future_value + sell_money - current_value
            reward[:, previous_action, current_action] = reward_scale * r

    for i in range(len(df) - 2, -1, -1):
        q_table[i] = reward[i] + gamma * q_table[i + 1].max(axis=1)
    return q_table
