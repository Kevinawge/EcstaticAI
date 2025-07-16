import pandas as pd
import numpy as np


class Backtester:
    def __init__(self, price_data: pd.DataFrame, signal_column: str, transaction_cost: float = 0.001):
        self.data = price_data.copy()
        self.signal_column = signal_column
        self.transaction_cost = transaction_cost
        self.results = None

    def run(self, initial_capital: float = 100000):
        print(
            f"[Backtester] Running backtest with capital = ${initial_capital:,.2f} and TC = {self.transaction_cost*100:.2f}%")
        df = self.data.copy()


        df["returns"] = df["Close"].pct_change()
        df["position"] = df[self.signal_column].shift()

        df["trade"] = df["position"].diff().abs()
        df["strategy_returns"] = df["position"] * df["returns"]
        df["strategy_returns"] -= df["trade"] * self.transaction_cost

        df["portfolio_value"] = (
            1 + df["strategy_returns"]).cumprod() * initial_capital
        df["cumulative_market"] = (
            1 + df["returns"]).cumprod() * initial_capital

        self.results = df.dropna()
        return self.results

    def performance_metrics(self):
        df = self.results.copy()

        total_return = df["portfolio_value"].iloc[-1] / \
            df["portfolio_value"].iloc[0] - 1
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = df["strategy_returns"].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else np.nan
        drawdown = (df["portfolio_value"] / df["portfolio_value"].cummax()) - 1
        max_drawdown = drawdown.min()

        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown
        }

    def summary(self):
        metrics = self.performance_metrics()
        print("\n--- Backtest Summary ---")
        for key, val in metrics.items():
            print(f"{key}: {val:.2%}")
        return metrics


#Test Block
if __name__ == "__main__":
    from alpha_model import AlphaModel

    alpha = AlphaModel()
    momentum_df = alpha.momentum_strategy()

    print("\n[Testing Backtester on Momentum Strategy]")
    backtest = Backtester(
        momentum_df, signal_column="signal_momentum", transaction_cost=0.001)
    results = backtest.run()
    backtest.summary()

    print("\n[Preview of Backtest Results]")
    print(results[["Close", "strategy_returns", "portfolio_value"]].tail())
