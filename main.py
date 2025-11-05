# main.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Dict, Tuple

from signal_generator import fetch_close_series, build_alpha
from risk_manager import ExecParams, simulate_trade_path
from pathlib import Path

# storage paths that match your repo layout
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
LOG_DIR = STORAGE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

from notification import notify
import yfinance as yf  # for SPY benchmark

# ---------- metrics ----------
def _metrics(equity: pd.Series) -> Dict[str, float]:
    ret = equity.pct_change().dropna()
    if ret.empty: 
        return {"CAGR":0, "Sharpe":0, "MaxDrawdown":0, "Calmar":np.nan}
    years = len(ret)/252.0
    cagr = (equity.iloc[-1]/equity.iloc[0])**(1/years)-1
    sharpe = (ret.mean()/ret.std())*np.sqrt(252) if ret.std()!=0 else 0.0
    cum = (1+ret).cumprod()
    mdd = (cum/cum.cummax()-1).min()
    calmar = cagr/abs(mdd) if mdd!=0 else np.nan
    return {"CAGR":float(cagr),"Sharpe":float(sharpe),"MaxDrawdown":float(mdd),"Calmar":float(calmar)}

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
    px_map = {s: fetch_close_series(s, start_date, end_date, interval) for s in stock_list}
    # align dates across all symbols
    common_idx = None
    for s in stock_list:
        common_idx = px_map[s].index if common_idx is None else common_idx.intersection(px_map[s].index)
    px_map = {s: px_map[s].reindex(common_idx).dropna() for s in stock_list}

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

    # ---- benchmark (SPY) normalised to same initial capital ----
    bench_curve = None
    if compare_spy:
        spy = yf.Ticker("SPY").history(start=start_date, end=end_date, interval=interval, auto_adjust=True)["Close"]
        spy = spy.reindex(equity.index).dropna()
        bench_curve = (spy / spy.iloc[0]) * initial_capital
        # align again to be safe
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
    start_date = "2015-01-01"
    end_date   = "2019-12-31"
    stock_list = ["META","AAPL","AMZN","NFLX","GOOG"]  # MAANG

    out = run_backtest(
        start_date=start_date, end_date=end_date, stock_list=stock_list,
        ma_window=130, trailing_window=10, trailing_pct=0.93, short_threshold=0.02,
        atr_win=14, cooldown_bars=3, reentry_eps=0.005, min_hold=5, weekly_exec=True,
        k_atr_stop=2.5, giveback_atr=2.0, time_stop=60,
        target_vol=0.10, gross_cap=1.0, roll_vol=20, perf_lookback=63,
        initial_capital=500_000.0, compare_spy=True, tag="strategy1"
    )
    print("Portfolio metrics:", out["portfolio_metrics"])
