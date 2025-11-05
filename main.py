# main.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Dict, Tuple

from signal_generator import fetch_close_series, build_alpha
from risk_manager import ExecParams, simulate_trade_path
from pathlib import Path
import time

# storage paths that match your repo layout
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
LOG_DIR = STORAGE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

from notification import notify
import yfinance as yf  # for SPY benchmark

# ---------- metrics ----------
def _metrics(equity: pd.Series | pd.DataFrame) -> Dict[str, float]:
    # tolerate None/empty
    if equity is None:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDrawdown": 0.0, "Calmar": np.nan}

    # collapse any DataFrame to a single Series (first column if 1 col, else row-mean)
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0] if equity.shape[1] == 1 else equity.mean(axis=1)

    equity = equity.dropna()
    if equity.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDrawdown": 0.0, "Calmar": np.nan}

    ret = equity.pct_change().dropna()
    if ret.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDrawdown": 0.0, "Calmar": np.nan}

    years = len(ret) / 252.0 if len(ret) else 0.0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0

    # force scalars to avoid "truth value of a Series is ambiguous"
    std = float(ret.std())
    mean = float(ret.mean())
    sharpe = (mean / std) * np.sqrt(252) if std > 0 else 0.0

    cum = (1 + ret).cumprod()
    mdd = float((cum / cum.cummax() - 1).min()) if not cum.empty else 0.0
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan

    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDrawdown": mdd, "Calmar": calmar}

# ---------- rolling CAGR filter ----------
def _rolling_cagr(ser: pd.Series, win: int, ann: int = 252) -> pd.Series:
    # product of (1+r) over window → convert to CAGR
    growth = (1+ser).rolling(win).apply(lambda a: np.nanprod(a), raw=True)
    yrs = win/ann
    return growth.pow(1.0/yrs) - 1.0

# ---------- main backtest ----------
def run_backtest(
    start_date: str,
    end_date: str,
    stock_list: Iterable[str],
    # alpha params
    ma_window: int = 130,
    trailing_window: int = 10,
    trailing_pct: float = 0.93,
    short_threshold: float = 0.02,
    atr_win: int = 14,
    # execution/risk params
    cooldown_bars: int = 3,
    reentry_eps: float = 0.005,
    min_hold: int = 5,
    weekly_exec: bool = True,
    k_atr_stop: float = 2.5,
    giveback_atr: float = 2.0,
    time_stop: int = 60,
    # portfolio risk controls
    target_vol: float = 0.10,      # per-asset vol target (annualised)
    gross_cap: float = 1.0,        # sum |weights| cap
    roll_vol: int = 20,            # vol lookback (days)
    perf_lookback: int = 63,       # rolling CAGR filter window (~3m)
    # others
    initial_capital: float = 500_000.0,
    interval: str = "1d",
    tag: str | None = None,
    compare_spy: bool = True,
) -> Dict[str, object]:

    stock_list = list(stock_list)
    assert len(stock_list) > 0, "stock_list is empty"

    notify(f"Loading {len(stock_list)} symbols: {', '.join(stock_list)}")

    px_map = {}
    skipped = []

    for s in stock_list:
        try:
            px_map[s] = fetch_close_series(s, start_date, end_date, interval)
        except ValueError as e:
            skipped.append(s)
            notify(f"Skipping {s}: {e}")
        time.sleep(0.25)  # prevent rate limits from Yahoo

    if not px_map:
        raise ValueError("All symbols failed to load. Network or Yahoo is blocking requests.")
    for s, ser in px_map.items():
        notify(f"{s}: {ser.index.min().date()} → {ser.index.max().date()} ({len(ser)} rows)")

    # align dates across loaded symbols
    common_idx = None
    for s, ser in px_map.items():
        common_idx = ser.index if common_idx is None else common_idx.intersection(ser.index)

    # NEW: diagnostics + early failure if overlap is too small
    if common_idx is None or len(common_idx) == 0:
        raise ValueError(
            "No overlapping trading days across symbols. "
            "Reduce the ticker list or adjust the date range."
        )
    if len(common_idx) < 50:
        notify(f"Warning: very small overlapping date range across symbols: {len(common_idx)} days.")

    # reindex to the common intersection
    px_map = {s: ser.reindex(common_idx).dropna() for s, ser in px_map.items()}
    stock_list = list(px_map.keys())  # drop skipped symbols downstream
    if skipped:
        notify(f"Proceeding with {len(stock_list)} symbols after skipping: {', '.join(skipped)}")


    # alpha build + execution to per-symbol return streams
    exec_params = ExecParams(
        cooldown_bars=cooldown_bars, reentry_eps=reentry_eps, min_hold=min_hold,
        weekly_exec=weekly_exec, k_atr_stop=k_atr_stop, giveback_atr=giveback_atr, time_stop=time_stop
    )

    per_symbol = {}
    for s in stock_list:
        alpha = build_alpha(
            px_map[s], ma_window=ma_window, trailing_window=trailing_window,
            trailing_pct=trailing_pct, atr_win=atr_win, short_threshold=short_threshold
        )
        path = simulate_trade_path(alpha, exec_params)
        per_symbol[s] = pd.concat([alpha, path], axis=1)

    # collect per-symbol strategy returns & positions
    strat_ret = pd.DataFrame({s: per_symbol[s]["strat_ret"] for s in stock_list}).reindex(common_idx).fillna(0.0)
    positions = pd.DataFrame({s: per_symbol[s]["position"] for s in stock_list}).reindex(common_idx).fillna(0)

    # ---- volatility targeting (per-asset) + gross exposure cap ----
    ann = 252
    sigma = strat_ret.rolling(roll_vol).std() * np.sqrt(ann)
    raw_w = (positions.replace(0, np.nan) * (target_vol / sigma)).fillna(0.0)

    gross = raw_w.abs().sum(axis=1).replace(0, np.nan)
    scale = (gross_cap / gross).clip(upper=1.0)
    weights = (raw_w.T * scale).T.fillna(0.0)

    # ---- rolling performance filter (CAGR>0) ----
    rc = strat_ret.apply(lambda s: _rolling_cagr(s, win=perf_lookback))
    mask = (rc > 0).fillna(False)
    weights = weights.where(mask, 0.0)

    # re-scale after masking
    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    scale = (gross_cap / gross).clip(upper=1.0)
    weights = (weights.T * scale).T.fillna(0.0)

    # ---- portfolio aggregation ----
    port_ret = (strat_ret * weights).sum(axis=1).fillna(0.0)
    equity = (1 + port_ret).cumprod() * (initial_capital / (1 + port_ret).cumprod().iloc[0])
    if equity.empty:
        raise ValueError(
            "Equity curve is empty after aggregation. "
            "This usually means rolling windows + small overlap wiped all rows. "
            "Try a shorter perf_lookback/roll_vol or remove tickers with short histories."
        )

    # ---- benchmark (SPY) normalised to same initial capital ----
    bench_curve = None
    if compare_spy:
        spy_df = yf.download(
            "SPY", start=start_date, end=end_date, interval=interval,
            auto_adjust=True, progress=False, threads=False
        )
        if spy_df is not None and not spy_df.empty:
            spy_ser = spy_df["Close"] if "Close" in spy_df.columns else spy_df.iloc[:, 0]
            spy_ser = spy_ser.reindex(equity.index).dropna()
            if len(spy_ser) >= 2:
                bench_curve = (spy_ser / spy_ser.iloc[0]) * initial_capital
                # realign equity to benchmark dates to keep windows identical
                equity = equity.reindex(bench_curve.index).dropna()


    # ---- metrics ----
    port_metrics = _metrics(equity)
    bench_metrics = _metrics(bench_curve) if bench_curve is not None else None

    # ---- persist artefacts ----
    tag = tag or f"s1_{stock_list[0]}_{stock_list[-1]}_{ma_window}_{trailing_window}"

    # portfolio returns (pct change of equity), indexed by date
    returns_df = pd.DataFrame(
        {"portfolio": equity.pct_change().fillna(0.0)}
    )
    returns_path = STORAGE_DIR / "returns.csv"
    returns_df.to_csv(returns_path)

    # per-symbol positions & strategy returns (wide), indexed by date
    trades = pd.DataFrame(
        {**{f"pos_{s}": per_symbol[s]["position"] for s in stock_list},
         **{f"ret_{s}": per_symbol[s]["strat_ret"] for s in stock_list}}
    )
    trades_path = STORAGE_DIR / "trades.csv"
    trades.to_csv(trades_path)

    notify(f"Saved returns → {returns_path}")
    notify(f"Saved trades  → {trades_path}")


    return {
        "equity_curve": equity,                 # pd.Series
        "portfolio_metrics": port_metrics,      # dict
        "benchmark_curve": bench_curve,         # pd.Series | None
        "benchmark_metrics": bench_metrics,     # dict | None
        "weights": weights,                     # pd.DataFrame
        "positions": positions,                 # pd.DataFrame
        "per_symbol": per_symbol,               # dict[symbol] -> pd.DataFrame
    }

# --- smoke test (optional) ---
if __name__ == "__main__":
    start_date = "2010-01-01"
    end_date   = "2019-12-31"
    stock_list = ["NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","TSLA","GOOG","BRK-B"]

    out = run_backtest(
        start_date=start_date, end_date=end_date, stock_list=stock_list,
        ma_window=130, trailing_window=10, trailing_pct=0.93, short_threshold=0.02,
        atr_win=14, cooldown_bars=3, reentry_eps=0.005, min_hold=5, weekly_exec=True,
        k_atr_stop=2.5, giveback_atr=2.0, time_stop=60,
        target_vol=0.10, gross_cap=1.0, roll_vol=20, perf_lookback=63,
        initial_capital=500_000.0, compare_spy=True, tag="strategy1"
    )
    print("Portfolio metrics:", out["portfolio_metrics"])
