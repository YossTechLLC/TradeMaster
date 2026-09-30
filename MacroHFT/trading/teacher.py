"""[TradeMaster] Dynamic-programming Q-table teacher for the venue-aware env (fees, funding, turnover penalty)."""
import numpy as np

from MacroHFT.trading.config import TradingConfig


def _fee_rate(fees, C, cs):
    """Vectorised FeeModel.rate: fee per unit of notional at each price."""
    notional = cs * C
    if fees.model == "per_contract":
        return fees.taker / notional
    return np.maximum(fees.taker * notional, fees.min_per_contract) / notional


def make_q_table_trading(close: np.ndarray, cfg: TradingConfig, bar_hours: float, gamma: float = 0.99) -> np.ndarray:
    """Optimal action values, shape (T, n_action, n_action); q[i, p, c] = value of taking c at row i after p."""
    C = np.asarray(close, dtype=np.float64)
    T, n, v = len(C), cfg.n_action, cfg.venue
    d = np.asarray(cfg.directions)
    q = np.zeros((T, n, n))
    if T < 2:
        return q
    Ci = C[:-1]
    r = C[1:] / Ci - 1
    cost = _fee_rate(v.fees, Ci, v.contract_size) + cfg.turnover_penalty_bps * 1e-4
    fund = v.funding.rate_per_hour * bar_hours if v.funding is not None else 0.0
    e = np.full(T - 1, cfg.leverage) if cfg.sizing == "equity" else cfg.fixed_units * Ci / cfg.capital
    switch = np.abs(d[None, :] - d[:, None])                      # [p, c]
    gross = d[None, None, :] * r[:, None, None] - fund * d[None, None, :]
    reward = cfg.reward_scale * e[:, None, None] * (gross - cost[:, None, None] * switch[None])
    for i in range(T - 2, -1, -1):
        q[i] = reward[i] + gamma * q[i + 1].max(axis=1)[None, :]
    return q
