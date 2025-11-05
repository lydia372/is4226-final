# Trading System Skeleton

Starter layout for a systematic trading / backtesting project. It loads prices via
`yfinance`, builds signals (MA + trailing stop), applies risk controls, and saves
returns/trades + logs to `storage/`.

## Project Structure
- `main.py` - orchestrates the backtest (`run_backtest`) and optional SPY benchmark
- `signal_generator.py` - prepares price series and computes indicators/signals
- `risk_manager.py` - position sizing (vol target), stops, PnL & equity curve
- `notification.py` - file outputs (CSV) and console notices
- `storage/` - outputs: `returns.csv`, `trades.csv`, and `logs/run_backtest.log`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Use
Run provided notebook (`Group3_FinalTerm.ipynb`) [RECOMMENDED]

The final cell (for graders) defines:
- `start_date`,`end_date`,`stock_list`
- all strategy params
- calls `run_backtest(...)`
- renders visuals and prints metrics

### Quick Start
Run a simple in-sample backtest from the CLI:
```bash
python main.py
```

## Outputs
- CSVs
    - `storage/returns.csv` - daily returns / equity
    - `storage/trades.csv` - executed trades
- Logs 
    - `storage/logs/run_backtest.log` - run metadata and summary metrics

## Troubleshooting

- `ValueError: Data must be 1-dimensional`
Ensure you squeeze yfinance columns to a Series (handled in signal_generator.fetch_close_series)
- Benchmark misalignment
We reindex/align calendars; however if the error occurs please check your date range.
- `IndexError: single positional indexer is out-of-bounds` (i.e. NAN data from yfinance)
Please run:

```bash
pip uninstall -y yfinance
pip install yfinance
```

and restart your kernel for the notebook to rerun.

Thank you! - Group 3
