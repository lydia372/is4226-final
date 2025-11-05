# risk_manager.py
from __future__ import annotations
import pandas as pd
import numpy as np

class ExecParams:
    def __init__(
        self,
        cooldown_bars: int = 3,
        reentry_eps: float = 0.005,
        min_hold: int = 5,
        weekly_exec: bool = True,
        k_atr_stop: float = 2.5,   # catastrophic stop (× ATR_px)
        giveback_atr: float = 2.0, # profit giveback (× ATR_px)
        time_stop: int = 60,       # max bars in trade
    ):
        self.cooldown_bars = cooldown_bars
        self.reentry_eps = reentry_eps
        self.min_hold = min_hold
        self.weekly_exec = weekly_exec
        self.k_atr_stop = k_atr_stop
        self.giveback_atr = giveback_atr
        self.time_stop = time_stop

def _is_trade_day(dt: pd.Timestamp, weekly_exec: bool) -> bool:
    return (dt.weekday() == 4) if weekly_exec else True  # Friday only if weekly

def simulate_trade_path(df: pd.DataFrame, p: ExecParams) -> pd.DataFrame:
    """
    Inputs df columns: Close, MA, MA_slope, TrailStop, ATR_px, desired
    Returns DataFrame with: position {+1,0,-1}, entry_px, cashflow_ret (strategy return)
    """
    close = df["Close"]; ma = df["MA"]; ts = df["TrailStop"]; slope = df["MA_slope"]; atr = df["ATR_px"]; want = df["desired"]

    pos = np.zeros(len(df), dtype=int)
    entry_px = np.full(len(df), np.nan)
    bars_held = 0
    cooldown = 0
    peak = trough = np.nan

    for i in range(1, len(df)):
        dt = df.index[i]
        trade_day = _is_trade_day(dt, p.weekly_exec)

        prev = pos[i-1]
        curr = prev
        price = close.iloc[i]
        ap = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 0.0

        # catastrophic stop
        cat_long  = prev>0 and price < (entry_px[i-1] - p.k_atr_stop*ap)
        cat_short = prev<0 and price > (entry_px[i-1] + p.k_atr_stop*ap)

        # giveback (track extremes since entry)
        if prev>0:
            peak = price if np.isnan(peak) else max(peak, price)
            gb_exit = price < (peak - p.giveback_atr*ap)
        elif prev<0:
            trough = price if np.isnan(trough) else min(trough, price)
            gb_exit = price > (trough + p.giveback_atr*ap)
        else:
            peak = trough = np.nan
            gb_exit = False

        time_exit = (prev!=0) and (bars_held >= p.time_stop)

        # not a scheduled trade day → only allow catastrophic stop
        if not trade_day and not cat_long and not cat_short:
            curr = prev
        else:
            # manage open positions
            if prev>0:  # long
                base_exit = (want.iloc[i] <= 0)
                if (base_exit and bars_held >= p.min_hold) or cat_long or gb_exit or time_exit:
                    curr = 0; cooldown = p.cooldown_bars

            elif prev<0:  # short
                base_exit = (want.iloc[i] >= 0)
                if (base_exit and bars_held >= p.min_hold) or cat_short or gb_exit or time_exit:
                    curr = 0; cooldown = p.cooldown_bars

            else:  # flat → entry
                if cooldown > 0:
                    cooldown -= 1
                    curr = 0
                else:
                    go_long  = (price > ma.iloc[i]*(1+p.reentry_eps)) or (price > ts.iloc[i]*(1+p.reentry_eps))
                    go_short = (price < ma.iloc[i]*(1-p.reentry_eps)) and (price < ts.iloc[i]*(1-p.reentry_eps)) and (slope.iloc[i] < 0)
                    if want.iloc[i] > 0 and go_long:
                        curr = 1; entry_px[i] = price; bars_held = 0; peak = trough = price
                    elif want.iloc[i] < 0 and go_short:
                        curr = -1; entry_px[i] = price; bars_held = 0; peak = trough = price
                    else:
                        curr = 0

        # disallow instant flip through flat without cooldown
        if (prev>0 and curr<0) or (prev<0 and curr>0):
            curr = 0; cooldown = p.cooldown_bars

        # update bars_held & carry entry_px forward
        if curr != 0:
            bars_held = bars_held + 1 if curr == prev else 1
            if np.isnan(entry_px[i]):
                entry_px[i] = entry_px[i-1] if not np.isnan(entry_px[i-1]) else close.iloc[i]
        else:
            bars_held = 0
            entry_px[i] = np.nan if curr == 0 else entry_px[i]

        pos[i] = curr

    # returns: yesterday's position × today's % change
    daily_ret = close.pct_change().fillna(0.0)
    strat_ret = pd.Series(pos, index=df.index, dtype=int).shift(1).fillna(0) * daily_ret

    out = pd.DataFrame({
        "position": pd.Series(pos, index=df.index, dtype=int),
        "entry_px": entry_px,
        "strat_ret": strat_ret
    })
    return out
