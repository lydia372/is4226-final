# signal_generator.py
from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf

# ---------- data ----------
def fetch_close_series(symbol: str, start: str, end: str, interval: str = "1d") -> pd.Series:
    tk = yf.Ticker(symbol)
    hist = tk.history(start=start, end=end, interval=interval, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No data for {symbol} in [{start}, {end}]")
    return hist["Close"].rename(symbol)

# ---------- indicators ----------
def compute_indicators(
    px: pd.Series,
    ma_window: int,
    trailing_window: int,
    trailing_pct: float,
    atr_win: int,
) -> pd.DataFrame:
    df = pd.DataFrame({"Close": px})
    df["MA"] = df["Close"].rolling(ma_window, min_periods=ma_window).mean()
    df["MA_slope"] = df["MA"].diff()
    df["TrailStop"] = df["Close"].rolling(trailing_window, min_periods=trailing_window).max() * trailing_pct

    # simple ATR proxy from % moves (keeps us data-light)
    r = df["Close"].pct_change().abs()
    df["ATR_px"] = r.rolling(atr_win).mean() * df["Close"]  # ATR expressed in price units
    return df.dropna()

# ---------- desired (trend) signal: +1 long, -1 short, 0 flat ----------
def desired_signal(
    df: pd.DataFrame,
    short_threshold: float = 0.02,  # must be >=2% below MA to short
) -> pd.Series:
    close, ma, ts, slope = df["Close"], df["MA"], df["TrailStop"], df["MA_slope"]

    want = np.where(
        (close > ma) | (close > ts),  # permissive long
        1,
        np.where(
            (close < ma * (1 - short_threshold)) & (close < ts) & (slope < 0),  # selective short
            -1,
            0,
        ),
    )
    return pd.Series(want, index=df.index, name="desired").astype(int)

# convenience: one-call alpha prep
def build_alpha(
    px: pd.Series,
    ma_window: int,
    trailing_window: int,
    trailing_pct: float,
    atr_win: int,
    short_threshold: float,
) -> pd.DataFrame:
    ind = compute_indicators(px, ma_window, trailing_window, trailing_pct, atr_win)
    ind["desired"] = desired_signal(ind, short_threshold=short_threshold)
    return ind
