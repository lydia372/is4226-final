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


# ---------- market regime detection ----------
def detect_market_regime(
    px: pd.Series,
    lookback: int = 20,
    vol_lookback: int = 20,
    bear_threshold: float = -0.05,  # -5% return over lookback = bear market
    high_vol_multiplier: float = 1.5,  # 1.5x median vol = high volatility
) -> pd.Series:
    """
    Detect bear market regime based on rolling returns and volatility.
    Returns: 1 for bear market, 0 for normal/bull market
    """
    returns = px.pct_change()
    
    # Rolling return over lookback period
    rolling_ret = (1 + returns).rolling(lookback).apply(lambda x: np.nanprod(x), raw=True) - 1
    
    # Rolling volatility
    rolling_vol = returns.rolling(vol_lookback).std() * np.sqrt(252)
    median_vol = rolling_vol.rolling(252).median()  # 1-year median vol
    
    # Bear market: negative returns AND high volatility
    is_bear = (rolling_ret < bear_threshold) | (
        (rolling_ret < 0) & (rolling_vol > median_vol * high_vol_multiplier)
    )
    
    return pd.Series(is_bear.astype(int), index=px.index, name="bear_market").fillna(0)

# ---------- indicators ----------
def compute_indicators(
    px: pd.Series,
    ma_window: int,
    trailing_window: int,
    trailing_pct: float,
    atr_win: int,
    bear_lookback: int = 20,
) -> pd.DataFrame:
    df = pd.DataFrame({"Close": px})
    df["MA"] = df["Close"].rolling(ma_window, min_periods=ma_window).mean()
    df["MA_slope"] = df["MA"].diff()
    df["TrailStop"] = df["Close"].rolling(trailing_window, min_periods=trailing_window).max() * trailing_pct

    # simple ATR proxy from % moves (keeps us data-light)
    r = df["Close"].pct_change().abs()
    df["ATR_px"] = r.rolling(atr_win).mean() * df["Close"]  # ATR expressed in price units
    
    # Market regime detection
    df["bear_market"] = detect_market_regime(px, lookback=bear_lookback)
    
    # Rolling volatility for position sizing
    returns = df["Close"].pct_change()
    df["volatility"] = returns.rolling(20).std() * np.sqrt(252)  # Annualized vol
    
    return df.dropna()

# ---------- desired (trend) signal: +1 long, -1 short, 0 flat ----------
def desired_signal(
    df: pd.DataFrame,
    short_threshold: float = 0.02,  # must be >=2% below MA to short (normal market)
    bear_short_threshold: float = 0.01,  # more aggressive shorting in bear markets (1% below MA)
) -> pd.Series:
    close = df["Close"]
    ma = df["MA"]
    ts = df["TrailStop"]
    slope = df["MA_slope"]
    bear_market = df.get("bear_market", pd.Series(0, index=df.index))
    
    # Dynamic short threshold: lower in bear markets for more aggressive shorting
    dynamic_short_thresh = np.where(bear_market > 0, bear_short_threshold, short_threshold)
    
    # Long signals: more restrictive in bear markets
    long_condition = (close > ma) | (close > ts)
    # In bear markets, require stronger signal (price must be above MA by more)
    long_condition = np.where(
        bear_market > 0,
        (close > ma * 1.01) & (close > ts),  # Require 1% above MA in bear markets
        long_condition
    )
    
    # Short signals: more aggressive in bear markets
    # Normal: close < MA*(1-threshold) AND close < TS AND slope < 0
    # Bear: close < MA*(1-threshold) OR (close < TS AND slope < 0) - more permissive
    short_condition_normal = (
        (close < ma * (1 - short_threshold)) & 
        (close < ts) & 
        (slope < 0)
    )
    short_condition_bear = (
        (close < ma * (1 - bear_short_threshold)) | 
        ((close < ts) & (slope < 0))
    )
    short_condition = np.where(bear_market > 0, short_condition_bear, short_condition_normal)
    
    want = np.where(
        long_condition,
        1,
        np.where(
            short_condition,
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
    bear_lookback: int = 20,
    bear_short_threshold: float = 0.01,
) -> pd.DataFrame:
    ind = compute_indicators(px, ma_window, trailing_window, trailing_pct, atr_win, bear_lookback)
    ind["desired"] = desired_signal(ind, short_threshold=short_threshold, bear_short_threshold=bear_short_threshold)
    return ind
