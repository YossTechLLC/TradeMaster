"""[TradeMaster] replay(): bar-by-bar account replay of a SignalStream under a PolicyConfig (spec section 10 order)."""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tm_risk.breach import deleverage, mark_bar
from tm_risk.types import AccountState, Decision

BAR_KEYS = ("equity", "equity_mark", "contracts", "exposure", "pnl", "fee", "slippage", "funding", "halted", "peak_ref",
            "dd_all", "dd_ref", "signal", "stop_price", "close")


@dataclass
class ReplayResult:
    decisions: list
    bars: dict
    summary: dict
    policy: object = None
    venue: object = None
    capital: float = 0.0
    bar_hours: float = 0.0
    info: dict = field(default_factory=dict)


class _RollingMax:
    """Max of the last `w` values (monotonic deque); w <= 0 means all-time."""

    def __init__(self, w):
        self.w, self.q, self.i = w, deque(), -1

    def push(self, x):
        self.i += 1
        while self.q and self.q[-1][1] <= x:
            self.q.pop()
        self.q.append((self.i, x))
        if self.w > 0:
            while self.q[0][0] <= self.i - self.w:
                self.q.popleft()
        return self.q[0][1]


def _sign(x):
    return (x > 0) - (x < 0)


def replay(stream, policy, venue, capital=10000.0, val_stream=None) -> ReplayResult:
    """Replay `stream` bar by bar. Order per bar k: 1 mark, 2 peaks, 3 ledger, 4 breach, 5 signal, 6 records.
    Decisions use bars <= k only; the last bar always exits. The identity path (checks False, sizing identity) needs
    no sizing / admission / episode / stops / ledger modules."""
    policy.validate(venue)
    N, cs = len(stream), venue.contract_size
    checks = policy.checks
    identity = (not checks) and policy.sizing == "identity"
    if not checks and policy.sizing != "identity":
        raise ValueError(f"policy '{policy.name}': checks=False supports only sizing 'identity'")
    if checks:
        from tm_risk import stops
        from tm_risk.admission import admit
        from tm_risk.episode import RiskEpisode
        from tm_risk.sizing import size_request
    ledger = None
    if checks and policy.sizing == "kelly":
        from tm_risk.ledger import SignalLedger
        ledger = SignalLedger(stream, venue, policy)
        if policy.ledger_seed == "val":
            if val_stream is None:
                raise ValueError(f"policy '{policy.name}': ledger_seed 'val' needs val_stream")
            ledger.seed(val_stream)
    bh = stream.bar_hours
    win = int(math.ceil(policy.dd_window_days * 24 / bh)) if policy.dd_window_days > 0 else 0
    roll = _RollingMax(win)
    st = AccountState(capital=capital, equity=capital, peak_all=capital, peak_ref=capital)
    bars = {key: np.zeros(N) for key in BAR_KEYS}
    decisions, episode, ep_start = [], None, 0
    n_uncovered = 0
    ts = stream.timestamp

    def fill(price, contracts_traded):
        """Fee + slippage (USD) of trading |contracts_traded| at `price`."""
        return (venue.fees.cost(contracts_traded, price, cs),
                policy.slippage_bps * 1e-4 * abs(contracts_traded) * cs * price)

    def note(action, k, s, **kw):
        d = Decision(k=k, timestamp=ts[k], signal=s, action=action, **kw)
        decisions.append(d)
        return d

    for k in range(N):
        p = float(stream.close[k])
        pnl = fee = slip = funding = 0.0
        # 1. mark
        if k >= 1 and st.contracts != 0:
            mk = mark_bar(st, stream, k, venue, policy)
            pnl, fee, slip, funding = mk.pnl, mk.fee, mk.slippage, mk.funding
            if mk.event:
                if episode is not None:
                    episode.record_close(mk.realised)
                note(mk.event, k, int(stream.direction[k]), contracts=-mk.side * mk.closed, reason=f"fill {mk.price:.6g}")
        st.bankrupt = st.equity <= 0
        bars["equity_mark"][k] = st.equity
        # 2. peaks
        st.peak_all = max(st.peak_all, st.equity)
        st.peak_ref = roll.push(st.equity)
        st.halted = bool(checks and st.equity < (1 - policy.dd_halt) * st.peak_ref)
        # 3. ledger
        if ledger is not None:
            ledger.update(k)
        # 4. breach
        if checks and st.contracts != 0:
            mk = deleverage(st, stream, k, venue, policy)
            if mk.event:
                fee, slip = fee + mk.fee, slip + mk.slippage
                if episode is not None:
                    episode.record_close(mk.realised)
                note("breach", k, int(stream.direction[k]), contracts=-mk.side * mk.closed, reason="exposure above cap band")
        # 5. signal
        covered = bool(stream.covered[k])
        s = int(stream.direction[k]) if covered and k < N - 1 else 0
        if not covered and k < N - 1:
            n_uncovered += 1
        if s < 0 and not venue.allows_short:
            s = 0
        if checks and s != st.blocked_direction:
            st.blocked_direction = 0
        if s == 0:
            if st.contracts != 0:
                f_, sl_ = fill(p, st.contracts)
                if checks and episode is not None:
                    episode.record_close(st.contracts * cs * (p - st.entry_price) - f_ - sl_)
                st.equity -= f_ + sl_
                fee, slip = fee + f_, slip + sl_
                note("exit", k, 0, contracts=-st.contracts)
                st.contracts, st.stop_price, st.open_risk = 0.0, 0.0, 0.0
            episode = None
        else:
            if checks and (episode is None or k - ep_start >= policy.episode_max_bars):
                episode, ep_start = RiskEpisode(policy, start_k=k, equity=st.equity), k
            if st.contracts != 0 and _sign(st.contracts) == s:
                note("hold", k, s, contracts=st.contracts)
            elif identity:
                tgt = s * math.floor(policy.margin_leverage * st.equity / (p * cs)) if st.equity > 0 else 0.0
                f_, sl_ = fill(p, tgt - st.contracts)
                act = "flip" if st.contracts != 0 else "enter" if tgt != 0 else "abstain"
                st.equity -= f_ + sl_
                fee, slip = fee + f_, slip + sl_
                note(act, k, s, contracts=tgt - st.contracts if act == "flip" else tgt,
                     wanted=abs(tgt) * cs * p, permitted=abs(tgt) * cs * p, reason="" if tgt else "no equity")
                st.contracts, st.entry_price, st.stop_price = float(tgt), p if tgt else 0.0, 0.0
            else:
                flip = st.contracts != 0
                old_contracts = st.contracts
                if flip:                                    # close the opposite leg first; size against the state after it
                    f_, sl_ = fill(p, st.contracts)
                    episode.record_close(st.contracts * cs * (p - st.entry_price) - f_ - sl_)
                    st.equity -= f_ + sl_
                    fee, slip = fee + f_, slip + sl_
                    st.contracts, st.stop_price, st.open_risk = 0.0, 0.0, 0.0
                f, _info = size_request(k, s, st, stream, ledger, policy)
                d = admit(k, s, f, st, stream, policy, venue, episode)
                d.k, d.timestamp, d.signal, d.exposure_req = k, ts[k], s, f
                if f <= 0 and not d.vetoes:                 # the sizing arm itself declined (kelly abstain / no vol)
                    why = str(_info.get("reason", "zero exposure"))
                    d.binding, d.reason = "sizing", "sizing: " + ("insufficient support" if why.startswith("support") else why)
                if d.contracts > 0:
                    f_, sl_ = fill(p, d.contracts)
                    st.equity -= f_ + sl_
                    fee, slip = fee + f_, slip + sl_
                    st.contracts, st.entry_price = s * float(d.contracts), p
                    sd = stops.stop_distance(stream, k, policy) if policy.stop_atr_mult > 0 else 0.0
                    st.stop_price = p * (1 - s * sd) if sd > 0 else 0.0
                    st.open_risk = d.contracts * cs * p * stops.stressed_loss_per_unit(stream, k, venue, policy)
                    d.action = "flip" if flip else "enter"
                    d.contracts = st.contracts
                else:
                    if flip:                                # the old leg was closed and paid for: record that close
                        note("exit", k, s, contracts=-old_contracts, reason="flip: new leg refused")
                    d.action = "abstain"
                decisions.append(d)
        # 6. records
        e = st.equity
        for key, v in (("equity", e), ("contracts", st.contracts), ("exposure", abs(st.contracts) * cs * p / e if e > 0 else 0.0),
                       ("pnl", pnl), ("fee", fee), ("slippage", slip), ("funding", funding), ("halted", float(st.halted)),
                       ("peak_ref", st.peak_ref), ("dd_all", min(1.0, 1 - e / st.peak_all) if st.peak_all > 0 else 0.0),
                       ("dd_ref", min(1.0, 1 - e / st.peak_ref) if st.peak_ref > 0 else 0.0), ("signal", s),
                       ("stop_price", st.stop_price), ("close", p)):
            bars[key][k] = v
    info = dict(n_uncovered=n_uncovered, truncated=bool(stream.source.get("truncated", False)))
    if ledger is not None:                                  # raw-signal outcomes visible at the end (account independent)
        info["ledger"] = {side: ledger.outcomes(side) for side in (1, -1)}
    res = ReplayResult(decisions, bars, {}, policy, venue, capital, bh, info)
    try:
        from tm_risk.report import summarize
    except ImportError:                                     # report.py is added later; keep a minimal summary
        res.summary = dict(net_pnl=float(bars["equity"][-1] - capital) if N else 0.0,
                           final_equity=float(bars["equity"][-1]) if N else capital, bars=N)
    else:
        res.summary = summarize(res, capital, bh)
    return res
