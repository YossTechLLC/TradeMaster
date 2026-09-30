"""[TradeMaster] Turnover penalty: training-only shaping cost on traded notional, never part of equity."""


def turnover_penalty(traded_notional_usd: float, bps: float) -> float:
    return bps * 1e-4 * traded_notional_usd
