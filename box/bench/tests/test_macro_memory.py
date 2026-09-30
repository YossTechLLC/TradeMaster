"""Differential test: vectorised episodicmemory.query == the upstream per-entry loop, bit for bit."""
import os, sys
import numpy as np
TM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, TM)
from MacroHFT.RL.util.memory import episodicmemory, custom_kernel


def upstream_query(mem, h, action):  # verbatim copy of the original loop
    kernel_values = np.array([custom_kernel(h, hs) for hs in mem.buffer["hidden_state"]])
    top_k_indices = np.argsort(kernel_values)[-mem.k:]
    top_k_actions = mem.buffer["action"][top_k_indices]
    top_k_q_values = mem.buffer["q_value"][top_k_indices]
    mask = (top_k_actions == action).astype(float)
    weights = kernel_values[top_k_indices] / np.sum(kernel_values[top_k_indices])
    masked_weights = weights * mask
    normalized_weights = masked_weights / np.sum(masked_weights)
    return np.dot(normalized_weights, top_k_q_values)


def main():
    rng = np.random.default_rng(0)
    np.seterr(all="ignore")
    mem = episodicmemory(4320, 5, 36, 9, 64, "cpu")
    assert np.isnan(mem.query(rng.normal(size=(1, 64)), 0))  # not full yet -> nan
    for i in range(4320):
        mem.add(rng.normal(size=(1, 64)), int(rng.integers(2)), rng.normal(size=(1,)), rng.normal(size=(1, 36)),
                rng.normal(size=(1, 9)), int(rng.integers(2)))
    mem.buffer["hidden_state"][7] = mem.buffer["hidden_state"][9]  # a tie
    n_nan = 0
    for i in range(3000):
        h = rng.normal(size=(1, 64)) * rng.choice([0.01, 1, 5]) if i % 3 else mem.buffer["hidden_state"][rng.integers(4320)][None] + 1e-4
        a = int(rng.integers(2))
        new, old = mem.query(h, a), upstream_query(mem, h, a)
        assert (np.isnan(new) and np.isnan(old)) or new == old, (i, new, old)
        n_nan += np.isnan(new)
    print(f"memory.query: 3000 queries bit-identical ({n_nan} nan fallbacks)")


if __name__ == "__main__":
    main()
