"""[TradeMaster] Venue-aware MacroHFT environments (long/short, fees, funding, leverage, turnover penalty) and the env factory."""
import numpy as np
import gym
from gym import spaces

from MacroHFT.trading.accounting import Account
from MacroHFT.trading.bars import infer_bar_hours
from MacroHFT.trading.config import TradingConfig
from MacroHFT.trading.teacher import make_q_table_trading


class VenueTestingEnv(gym.Env):
    penalised = False   # the turnover penalty shapes training rewards only

    def __init__(self, df, tech_indicator_list, tech_indicator_list_trend, cfg: TradingConfig, clf_list=None,
                 back_time_length=1, initial_action=None, bar_hours=None):
        assert not cfg.legacy, "VenueTestingEnv needs a non-legacy TradingConfig; use make_env for legacy"
        self.cfg = cfg
        self.df = df
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.clf_list = clf_list
        self.initial_action = cfg.flat_action if initial_action is None else initial_action
        self.action_space = spaces.Discrete(cfg.n_action)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf,
                                            shape=(back_time_length * len(tech_indicator_list),))
        self.stack_length = back_time_length
        self.m = back_time_length
        self._single = df[tech_indicator_list].to_numpy()
        self._trend = df[tech_indicator_list_trend].to_numpy()
        self._clf = df[clf_list].to_numpy() if clf_list is not None else None
        self._close = df["close"].to_numpy()
        self._timestamp = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.arange(len(df))
        lo = "raw_low" if "raw_low" in df.columns else "low" if "low" in df.columns else None
        hi = "raw_high" if "raw_high" in df.columns else "high" if "high" in df.columns else None
        self._low = df[lo].to_numpy() if lo and hi else None
        self._high = df[hi].to_numpy() if lo and hi else None
        self._n_rows = len(df.index.unique())
        self.bar_hours = infer_bar_hours(self._timestamp) if bar_hours is None else bar_hours
        self.account = Account(cfg, self.bar_hours)
        self.terminal = False
        self.previous_action = self.initial_action
        self.reward_history = [0]

    def _states(self):
        self.single_state = self._single[self.m - self.stack_length:self.m]
        self.trend_state = self._trend[self.m - self.stack_length:self.m]
        if self._clf is not None:
            self.clf_state = self._clf[self.m - self.stack_length:self.m]

    def _pack(self, *rest):
        if self._clf is None:
            return (self.single_state, self.trend_state) + rest
        return (self.single_state, self.trend_state, self.clf_state.reshape(-1)) + rest

    def reset(self):
        self.terminal = False
        self.m = self.stack_length
        self._states()
        self.previous_action = self.initial_action
        self.reward_history = [0]
        self.account.reset(self.cfg.directions[self.initial_action], self._close[self.m - 1])
        return self._pack({"previous_action": self.initial_action})

    def step(self, action):
        m0 = self.m
        self.m += 1
        self._states()
        low = self._low[self.m - 1] if self._low is not None else None
        high = self._high[self.m - 1] if self._high is not None else None
        res = self.account.step(self.cfg.directions[action], self._close[m0 - 1], self._close[self.m - 1], low, high)
        self.reward = float((res.pnl - res.fee - res.funding - res.penalty * self.penalised) / self.cfg.capital * self.cfg.reward_scale)
        self.reward_history.append(self.reward)
        self.previous_action = action
        self.terminal = (m0 >= self._n_rows - 1) or bool(res.bankrupt)
        if self.terminal:
            self.final_balance = self.account.equity - self.cfg.capital
            self.required_money = self.cfg.capital
            self.pured_balance = self.final_balance
            self.trading_log = dict(self.account.log(), close=self._close[self.stack_length:self.m],
                                    close_start=self._close[self.stack_length - 1],
                                    timestamp=self._timestamp[self.stack_length:self.m])
            print("the portfit margine is ", self.final_balance / self.required_money)
        return self._pack(self.reward, self.terminal, {"previous_action": action})

    def get_final_return_rate(self, slient=False):
        final_balance = self.account.equity - self.cfg.capital
        fee = float(np.sum([r.fee for r in self.account._log]))
        return final_balance / self.cfg.capital, final_balance, self.cfg.capital, fee


class VenueTrainingEnv(VenueTestingEnv):
    penalised = True

    def __init__(self, df, tech_indicator_list, tech_indicator_list_trend, cfg: TradingConfig, clf_list=None,
                 back_time_length=1, initial_action=None, bar_hours=None):
        super().__init__(df, tech_indicator_list, tech_indicator_list_trend, cfg, clf_list, back_time_length,
                         initial_action, bar_hours)
        self.q_table = make_q_table_trading(self._close, cfg, self.bar_hours)

    def reset(self):
        out = super().reset()
        out[-1]["q_value"] = self.q_table[self.m - 1][self.previous_action][:]
        return out

    def step(self, action):
        out = super().step(action)
        out[-1]["q_value"] = self.q_table[self.m - 1][action][:]
        return out


def make_env(kind, level, df, tech_indicator_list, tech_indicator_list_trend, cfg, clf_list=None,
             transcation_cost=0.0002, back_time_length=1, max_holding_number=0.01, initial_action=0):
    """Legacy classes (exact upstream arguments) when cfg.legacy, else the venue-aware envs."""
    assert kind in ("train", "test") and level in ("low", "high"), (kind, level)
    if cfg.legacy:
        # imported lazily: the legacy modules read ./data/feature_list at import time
        if level == "low":
            from MacroHFT.env.low_level_env import Testing_Env, Training_Env
            kw = dict(df=df, tech_indicator_list=tech_indicator_list,
                      tech_indicator_list_trend=tech_indicator_list_trend, transcation_cost=transcation_cost,
                      back_time_length=back_time_length, max_holding_number=max_holding_number,
                      initial_action=initial_action)
        else:
            from MacroHFT.env.high_level_env import Testing_Env, Training_Env
            kw = dict(df=df, tech_indicator_list=tech_indicator_list,
                      tech_indicator_list_trend=tech_indicator_list_trend, clf_list=clf_list,
                      transcation_cost=transcation_cost, back_time_length=back_time_length,
                      max_holding_number=max_holding_number, initial_action=initial_action)
        return Training_Env(alpha=0, **kw) if kind == "train" else Testing_Env(**kw)
    cls = VenueTrainingEnv if kind == "train" else VenueTestingEnv
    return cls(df, tech_indicator_list, tech_indicator_list_trend, cfg, clf_list=clf_list if level == "high" else None,
               back_time_length=back_time_length, initial_action=initial_action)
