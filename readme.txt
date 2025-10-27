Trading System Skeleton
=======================

This repository provides a starter layout for a systematic trading or backtesting project. The modules are intentionally light so you can layer in your preferred data sources, alpha ideas, and execution logic.

Project Structure
-----------------
- `main.py` – Orchestrates the core backtesting / live iteration loop.
- `signal_generator.py` – Builds features and trading signals from raw market data.
- `risk_manager.py` – Applies risk controls (position sizing, stop losses, pnl updates).
- `notification.py` – Handles logging and persistence of trades/returns.
- `storage/` – Default persistence path for CSV exports and logs.

Getting Started
---------------
1. Populate `storage/` with your historical or live data inputs.
2. Flesh out `SignalGenerator.prepare_data` and `SignalGenerator.generate_signals` with your alpha logic.
3. Implement the sizing and pnl logic inside `RiskManager` to match your strategy constraints.
4. Extend `BacktestingEngine.load_data` and `BacktestingEngine.execute_orders` to wire in real data feeds / broker APIs.
5. Run `python main.py` to execute a single iteration (looping/backtest driver is up to you).

Next Steps
----------
- Add tests (e.g., `pytest`) to validate the signal and risk modules.
- Introduce configuration management (`yaml`, `.env`, CLI args) to manage parameter sweeps.
- Replace the placeholder notification logic with your analytics stack or alerting system.
