"""[TradeMaster] Shared bar-spacing inference used by the environments and the report."""
import numpy as np
import pandas as pd

DEFAULT_BAR_HOURS = 1 / 60


def infer_bar_hours(timestamp, default=DEFAULT_BAR_HOURS):
    """Median bar spacing in hours; `default` when the timestamps are not real datetimes."""
    ts = np.asarray(timestamp)
    if len(ts) < 2 or ts.dtype.kind not in "MOU":
        return default
    try:
        ns = pd.to_datetime(pd.Series(ts)).to_numpy().astype("datetime64[ns]").astype(np.int64)
    except (ValueError, TypeError):
        return default
    step = float(np.median(np.diff(ns))) / 3.6e12
    return step if np.isfinite(step) and step > 0 else default
