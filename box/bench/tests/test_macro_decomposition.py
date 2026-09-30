"""rolling_slope (dot-product form) == upstream rolling().apply(get_slope_window) to float rounding."""
import os, sys, time
import numpy as np, pandas as pd
TM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(TM, "MacroHFT", "preprocess"))
from decomposition import rolling_slope, get_slope_window

close = pd.read_feather(os.path.join(TM, "MacroHFT/data/BTCUSDT/df_train.feather"))["close"][:4000]
t = time.time(); old = close.rolling(window=360).apply(get_slope_window).to_numpy(); t_old = time.time() - t
t = time.time(); new = rolling_slope(close, 360); t_new = time.time() - t
assert np.array_equal(np.isnan(old), np.isnan(new))
ok = ~np.isnan(old)
err = np.max(np.abs(new[ok] - old[ok]) / (np.abs(old[ok]).max()))
assert err < 1e-9, err
print(f"rolling_slope: {ok.sum()} windows, max error {err:.1e} of the slope scale; upstream {t_old:.2f} s, new {t_new:.3f} s")
