"""[TradeMaster] SignalLedger: outcomes of the raw model signal at unit notional, independent of any account state."""
import numpy as np


def signal_episodes(stream, venue, policy):
    """Episodes of `stream`: list of (exit_bar, side, r, hours, sigma_entry) ordered by exit bar (spec section 4)."""
    from tm_risk.stops import bar_vol
    d = np.where(stream.covered, stream.direction, 0).astype(int)
    c = stream.close
    fee = lambda p: venue.fees.rate(float(p), venue.contract_size)
    fund = venue.funding.rate_per_hour if venue.funding is not None else 0.0
    out, n, s = [], len(d), 0
    while s < n:
        if d[s] == 0:
            s += 1
            continue
        e = s + 1
        while e < n and d[e] == d[s]:
            e += 1
        if e < n:                                   # exit bar e = first row with a different direction
            side, hours = int(d[s]), (e - s) * stream.bar_hours
            r = side * (c[e] / c[s] - 1) - fee(c[s]) - fee(c[e]) - side * fund * hours
            out.append((e, side, float(r), float(hours), float(bar_vol(stream, s, policy.vol_window_bars))))
        s = e
    return out


class SignalLedger:
    """Closed signal episodes visible at bar k (exit bar <= k). Takes only the stream, venue and policy fees/windows."""

    def __init__(self, stream, venue, policy):
        self._eps = signal_episodes(stream, venue, policy)
        self._ptr = 0
        self._rows = {1: [], -1: []}                # side -> [(r, hours, sigma)]
        self._venue, self._policy = venue, policy
        self.cache = {}                             # scratch for consumers (consumers key entries by side and n(side))

    def _add(self, side, r, h, sg):
        self._rows[side].append((r, h, sg))

    def update(self, k):
        """Append every episode whose exit bar is <= k (causal visibility)."""
        while self._ptr < len(self._eps) and self._eps[self._ptr][0] <= k:
            _, side, r, h, sg = self._eps[self._ptr]
            self._add(side, r, h, sg)
            self._ptr += 1

    def seed(self, stream_val):
        """Add every episode of a validation stream up front (selection-biased; see PolicyConfig.ledger_seed)."""
        for _, side, r, h, sg in signal_episodes(stream_val, self._venue, self._policy):
            self._add(side, r, h, sg)

    def outcomes(self, side):
        rows = self._rows[1 if side > 0 else -1]
        a = np.asarray(rows, dtype=float).reshape(-1, 3)
        return a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()

    def n(self, side):
        return len(self._rows[1 if side > 0 else -1])
