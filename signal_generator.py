# signal_generator.py
from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
import time

# ---------- data ----------
def _dl(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    # threads=False is important in notebooks / CI
    return yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

def fetch_close_series(symbol: str, start: str, end: str, interval: str = "1d") -> pd.Series:
    """
    Robust loader:
    1) Try bounded download.
    2) If empty, retry a few times (short sleeps).
    3) If still empty, try period='max' then slice.
    4) If still empty, raise ValueError.
    """
    # --- attempt 1..3: normal window ---
    df = None
    for attempt in range(3):
        try:
            df = _dl(symbol, start, end, interval)
            if df is not None and not df.empty:
                break
        except Exception as e:
            # mild backoff
            time.sleep(0.8 * (attempt + 1))
        time.sleep(0.4)

    # --- fallback: period=max then trim ---
    if df is None or df.empty:
        try:
            df = yf.download(
                tickers=symbol, period="max", interval=interval,
                auto_adjust=True, progress=False, threads=False,
            )
            if df is not None and not df.empty:
                df = df.loc[
                    (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))
                ]
        except Exception:
            pass

    if df is None or df.empty:
        raise ValueError(f"No data for {symbol} in [{start}, {end}]")

    # columns now single-level; prefer "Close", fallback to first column
    if "Close" in df.columns:
        ser = df["Close"]
    else:
        ser = df.iloc[:, 0]

    # --- ensure 1D: sometimes yfinance gives (n,1) DataFrame ---
    if isinstance(ser, pd.DataFrame):
        ser = ser.squeeze("columns")

    ser = ser.astype(float)
    if getattr(ser.index, "tz", None) is not None:
        ser.index = ser.index.tz_localize(None)
    ser = ser.sort_index().dropna()
    ser.name = symbol
    assert isinstance(ser, pd.Series), f"Expected Series, got {type(ser)} for {symbol}"
    return ser


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
