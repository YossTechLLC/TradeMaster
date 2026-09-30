"""[TradeMaster] SignalStream: bars plus a model's desired direction per bar close, with loaders (model-agnostic)."""
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_BAR_HOURS = 1 / 60


@dataclass
class SignalStream:
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray            # USD per bar; zeros if unknown (participation then disabled)
    direction: np.ndarray               # int8 in {-1,0,1}: desired position decided at the CLOSE of bar k, held over k+1
    covered: np.ndarray                 # bool: the model produced a decision at k (False -> treated as 0, counted)
    bar_hours: float
    source: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.close)

    def head(self, n) -> "SignalStream":
        """The first n bars (a truncated stream; decisions on it must equal the prefix of the full replay)."""
        src = dict(self.source, truncated=True, n_covered=int(self.covered[:n].sum()))
        return SignalStream(*(getattr(self, f)[:n] for f in ("timestamp", "open", "high", "low", "close",
                                                             "quote_volume", "direction", "covered")),
                            bar_hours=self.bar_hours, source=src)


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


def _col(df, name):
    """raw_<name> when present, else <name>, else None."""
    for c in ("raw_" + name, name):
        if c in df.columns:
            return df[c].to_numpy(dtype=float)
    return None


def bar_arrays(df):
    """(timestamp, open, high, low, close, quote_volume) numpy arrays from a bar frame (see module rules)."""
    close = _col(df, "close")
    if close is None:
        raise ValueError("bar frame needs a 'close' column")
    ts = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.arange(len(df))
    o, h, l = _col(df, "open"), _col(df, "high"), _col(df, "low")
    if o is None:
        o = np.concatenate([close[:1], close[:-1]])
    if h is None:
        h = np.maximum(o, close)
    if l is None:
        l = np.minimum(o, close)
    qv = _col(df, "quote_volume")
    if qv is None:
        warnings.warn("no quote_volume column: participation limit disabled", stacklevel=3)
        qv = np.zeros(len(df))
    return ts, o, h, l, close, qv


def _stream(df, direction, covered, bar_hours, source):
    ts, o, h, l, c, qv = bar_arrays(df)
    bh = float(bar_hours) if bar_hours else infer_bar_hours(ts)
    direction = np.asarray(direction).astype(np.int8)
    if not np.isin(direction, (-1, 0, 1)).all():
        raise ValueError("direction must be in {-1, 0, 1}")
    return SignalStream(ts, o, h, l, c, qv, direction, np.asarray(covered, dtype=bool), bh, source)


def from_frame(df, direction_col="direction", covered_col=None, bar_hours=None, source=None):
    """Generic: any frame with timestamp + OHLC(+quote_volume) + a direction column (decided at each bar's close)."""
    d = df[direction_col].to_numpy(dtype=float)
    covered = df[covered_col].to_numpy(dtype=bool) if covered_col else ~np.isnan(d)
    src = dict(source or {}, n_covered=int(covered.sum()), truncated=False)
    return _stream(df, np.nan_to_num(d), covered, bar_hours, src)


def from_signal_npz(path, bars_df, timestamp_is_hold_bar=True, bar_hours=None):
    """npz with `timestamp`, `direction` (+`covered`) joined to `bars_df` on timestamp.

    Convention of MacroHFT trading logs and export_signal: the npz timestamp is the bar the position is HELD over, so the
    decision was taken at the previous bar's close (timestamp_is_hold_bar=True). With False the timestamp is the decision bar.
    """
    with np.load(path, allow_pickle=True) as z:
        ts, dr = z["timestamp"], np.asarray(z["direction"], dtype=float)
        cov = np.asarray(z["covered"], dtype=bool) if "covered" in z.files else np.ones(len(dr), dtype=bool)
    idx = pd.Index(pd.to_datetime(bars_df["timestamp"]))
    rows = idx.get_indexer(pd.to_datetime(pd.Series(ts)))
    if (rows < 0).any():
        raise ValueError(f"{path}: {int((rows < 0).sum())} timestamps not found in the bar frame")
    if timestamp_is_hold_bar:
        rows = rows - 1
        if rows.min() < 0:
            raise ValueError(f"{path}: first signal row has no preceding bar to decide at")
    n = len(bars_df)
    direction, covered = np.zeros(n), np.zeros(n, dtype=bool)
    direction[rows], covered[rows] = dr, cov
    truncated = bool(rows.max() < n - 2)
    src = dict(model="signal_npz", path=str(path), truncated=truncated, n_covered=int(covered.sum()))
    return _stream(bars_df, direction, covered, bar_hours, src)


def from_macrohft(run_dir, dataset_dir, split="test"):
    """MacroHFT run -> SignalStream over every bar of <dataset_dir>/whole/<split>.feather.

    Prefers <run_dir>/<split>/signal_<split>.npz (export_signal), else the trading log (trading_log.npz or
    trading_log_<split>.npz): log row i holds the position over df row stack+i, decided at df row stack-1+i. Joined on
    timestamp; asserts log close == df close. source["truncated"] is set if the log ends before the df.
    """
    df = pd.read_feather(os.path.join(dataset_dir, "whole", f"{split}.feather"))
    sdir = os.path.join(run_dir, split)
    sig = os.path.join(sdir, f"signal_{split}.npz")
    if os.path.isfile(sig):
        st = from_signal_npz(sig, df)
        st.source.update(model="macrohft", run_dir=str(run_dir), split=split, kind="signal_npz")
        return st
    for name in ("trading_log.npz", f"trading_log_{split}.npz"):
        path = os.path.join(sdir, name)
        if os.path.isfile(path):
            break
    else:
        raise FileNotFoundError(f"no signal_{split}.npz or trading log under {sdir}")
    with np.load(path, allow_pickle=True) as z:
        log = {k: z[k] for k in z.files}
    dr, lclose, n = np.asarray(log["direction"], dtype=float), np.asarray(log["close"], dtype=float), len(log["direction"])
    idx = pd.Index(pd.to_datetime(df["timestamp"]))
    r0 = idx.get_indexer(pd.to_datetime(pd.Series(np.asarray(log["timestamp"])[:1])))[0]
    if r0 < 1:
        raise ValueError(f"{path}: first log timestamp not found in {split}.feather (or has no preceding bar)")
    if r0 + n > len(df):
        raise ValueError(f"{path}: log ({n} rows from df row {r0}) is longer than the bar frame ({len(df)})")
    close = df["close"].to_numpy(dtype=float)
    if not np.allclose(lclose, close[r0:r0 + n], rtol=0, atol=1e-9):
        raise ValueError(f"{path}: log close differs from {split}.feather close at rows {r0}..{r0 + n - 1}")
    if "close_start" in log and not np.isclose(float(log["close_start"]), close[r0 - 1], rtol=0, atol=1e-9):
        raise ValueError(f"{path}: log close_start differs from df close at row {r0 - 1}")
    direction, covered = np.zeros(len(df)), np.zeros(len(df), dtype=bool)
    direction[r0 - 1:r0 - 1 + n], covered[r0 - 1:r0 - 1 + n] = dr, True
    src = dict(model="macrohft", run_dir=str(run_dir), split=split, kind=os.path.basename(path), stack=int(r0),
               truncated=bool(r0 + n < len(df)), n_covered=int(n))
    return _stream(df, direction, covered, None, src)
