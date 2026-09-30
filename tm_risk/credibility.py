"""[TradeMaster] Credibility shrinkage of episode outcomes toward zero edge."""
import numpy as np


def shrink(r, n0):
    """With c = n/(n+n0): r - (1-c)*mean(r). The mean moves toward zero, the residuals are kept."""
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return r
    c = len(r) / (len(r) + n0)
    return r - (1 - c) * r.mean()
