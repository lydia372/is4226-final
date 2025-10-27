import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator

class SignalGenerator:
    def __init__(self, symbol, start, end, interval, capital, transaction_cost, verbose = True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.initial_capital = capital
        self.capital = capital
        self.transaction_cost = transaction_cost
        self.quantity = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.prepare_data()
        

    def prepare_data(self):
        # since we are building a common class for all types of strategy, we will not calcualte the moving averages now.
        # we will calculate the returns though.
        # Since most strategies utilise close prices we are only factoring close price. However, you can alter acoordingly.
        stock_data = yf.Ticker(self.symbol)
        hist_stock = stock_data.history(start = self.start, end = self.end, interval = self.interval)
        bt_data = pd.DataFrame()
        bt_data["Close_Price"] = hist_stock["Close"]
        bt_data["Return"] = np.log(bt_data["Close_Price"] / bt_data["Close_Price"].shift(1))
        bt_data = bt_data.dropna()
        self.data = bt_data

    def close_graph(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.data["Close_Price"] ,color='black', label='Price', linestyle='dashed')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.title("Close Prices of {}".format (self.symbol))
        plt.legend()
        plt.grid()
        plt.show()

    def return_date_price(self, bar):
        # A bar is a unit of data at a given time, depends on the interval you choose, it provides you OHLCV and time info
        # Since we have modeled close prices, we will get the price and date
        date = str(self.data.index[bar])[:10]
        price = self.data.Close_Price.iloc[bar]
        return date, price
    
    def realised_balance(self, bar):
        """
        Returns the realised capital (actual cash) in your account
        at a given time period / bar.
        """
        date, price = self.return_date_price(bar)
        print("Date: {} | Realised Balance: {:.1f}".format(date, self.capital))


    def unrealised_balance(self, bar):
        """
        Returns the unrealised capital (open trades not yet settled)
        in your account at a given time period / bar.
        """
        date, price = self.return_date_price(bar)
        ub = self.quantity * price
        print("Date: {} | Unrealised Balance: {:.1f}".format(date, ub))


    def total_balance(self, bar):
        """
        Returns the total balance = realised + unrealised.
        """
        date, price = self.return_date_price(bar)
        tb = self.quantity * price + self.capital
        print("Date: {} | Total Balance: {:.1f}".format(date, tb))


    def buy_order(self, bar, quantity=None, dollar=None):
        """
        Executes a buy order.
        If `quantity` is not provided, it is calculated based on the `dollar` amount and current price.
        """
        date, price = self.return_date_price(bar)

        # Determine quantity
        if quantity is None:
            if dollar is None:
                raise ValueError("Either quantity or dollar amount must be provided for buy orders.")
            if isinstance(dollar, str) and dollar.lower() == "all":
                dollar = self.capital
            quantity = int(dollar / price)

        if quantity <= 0:
            if self.verbose:
                print("Buy order skipped: non-positive quantity.")
            return

        required_capital = (quantity * price) * (1 + self.transaction_cost)
        if required_capital > self.capital:
            if self.verbose:
                print("Buy order skipped: insufficient capital.")
            return

        # Deduct capital (include transaction cost)
        self.capital -= required_capital

        # Update holdings and trades
        self.quantity += quantity
        self.trades += 1

        # Logging
        if self.verbose:
            print("Bought {} shares of {} at {:.1f} per share worth {:.1f} $".format(
                quantity, self.symbol, price, quantity * price
            ))

            self.realised_balance(bar)
            self.unrealised_balance(bar)
            self.total_balance(bar)


    def sell_order(self, bar, quantity=None, dollar=None):
        """
        Executes a sell order.
        If `quantity` is not provided, it is calculated based on the `dollar` amount and current price.
        """
        date, price = self.return_date_price(bar)

        # Determine quantity
        if quantity is None:
            if dollar is None:
                raise ValueError("Either quantity or dollar amount must be provided for sell orders.")
            if isinstance(dollar, str) and dollar.lower() == "all":
                if self.quantity > 0:
                    dollar = self.quantity * price
                else:
                    dollar = self.capital
            quantity = int(dollar / price)

        if quantity <= 0:
            if self.verbose:
                print("Sell order skipped: non-positive quantity.")
            return

        # Add capital (minus transaction cost)
        self.capital += (quantity * price) * (1 - self.transaction_cost)

        # Update holdings and trades
        self.quantity -= quantity
        self.trades += 1

        # Logging
        if self.verbose:
            print("Sold {} shares of {} at {:.1f} per share worth {:.1f} $".format(
                quantity, self.symbol, price, quantity * price
            ))

            self.realised_balance(bar)
            self.unrealised_balance(bar)
            self.total_balance(bar)

    # Close any open position at the end of the backtesting
    def go_long(self, bar, dollar="all"):
        """
        Convenience wrapper to enter a long position using either dollar amount or all capital.
        """
        self.buy_order(bar, dollar=dollar)

    def go_short(self, bar, dollar="all"):
        """
        Convenience wrapper to enter a short position (or reduce longs) using provided capital proxy.
        """
        self.sell_order(bar, dollar=dollar)

    # Close any open position at the end of the backtesting
    def last_trade(self, bar):
        """
        Closes any open position at the end of backtesting,
        realizes final PnL, and prints a performance summary.
        """
        date, price = self.return_date_price(bar)
        last_quantity = self.quantity

        # Sell all remaining holdings (if any)
        self.capital += self.quantity * price
        self.quantity = 0  # all settled, no open position left
        self.trades += 1

        if self.verbose:
            print("Closed open trades for {} shares of {} at {:.1f} per share worth {:.1f} $".format(
                last_quantity, self.symbol, price, last_quantity * price
            ))

            # Show balances
            self.total_balance(bar)

        # Calculate and print final results
        returns = (self.capital - self.initial_capital) / self.initial_capital * 100

        print("\nThe total capital at end of strategy: {:.1f}".format(self.capital))
        print("The strategy returns on investment are {:.1f}%".format(returns))
        print("Total trades by strategy are {:.0f}".format(self.trades))


    def run_strategy(self, STMA_window, LTMA_window):
        self.position = 0
        self.trades = 0
        self.capital = self.initial_capital

        if len(self.data) <= LTMA_window:
            raise ValueError("Not enough data to compute the long-term moving average window.")

        # Calculate moving averages
        indicator_1 = SMAIndicator(close=self.data["Close_Price"], window=STMA_window, fillna=False)
        STMA = indicator_1.sma_indicator()

        indicator_2 = SMAIndicator(close=self.data["Close_Price"], window=LTMA_window, fillna=False)
        LTMA = indicator_2.sma_indicator()

        self.data["STMA"] = STMA
        self.data["LTMA"] = LTMA

        # Main loop
        for bar in range(LTMA_window, len(self.data)):
            # Check for LONG condition
            if self.position in [0, -1]:  # no position or short
                if self.data["STMA"].iloc[bar] > self.data["LTMA"].iloc[bar]:
                    self.go_long(bar, dollar="all")  # go with all money
                    self.position = 1  # long created
                    print("--")

            # Check for SHORT condition
            if self.position in [0, 1]:  # no position or long
                if self.data["STMA"].iloc[bar] < self.data["LTMA"].iloc[bar]:
                    self.go_short(bar, dollar="all")  # go with all money
                    self.position = -1  # short created
                    print("--")

        # Close all open positions
        last_bar = len(self.data) - 1
        self.last_trade(last_bar)