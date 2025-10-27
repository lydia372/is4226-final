from signal_generator import SignalGenerator

if __name__ == "__main__":
    A = SignalGenerator("AAPL", "2010-04-10", "2022-07-01", "1d", 10000, 0.01, True)
    A.run_strategy(STMA_window=50, LTMA_window=200)
