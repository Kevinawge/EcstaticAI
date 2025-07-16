import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class YFinanceFetcher:
    def __init__(self, ticker: str, period: str = "1y", interval: str = "1d", verbose: bool = True):
        self.ticker = ticker.upper()
        self.period = period
        self.interval = interval
        self.verbose = verbose
        self.data = self._download_data()

    def _download_data(self) -> pd.DataFrame:
        if self.verbose:
            print(
                f"[YF] Downloading {self.ticker} data for period '{self.period}' and interval '{self.interval}'")
        try:
            df = yf.download(self.ticker, period=self.period,
                             interval=self.interval)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data for {self.ticker}: {str(e)}")

    def get_price_data(self) -> pd.DataFrame:
        return self.data[["Open", "High", "Low", "Close", "Volume"]]

    def get_returns(self, log: bool = False) -> pd.Series:
        prices = self.data["Close"]
        if log:
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()
        return returns

    def get_recent_returns(self, days: int = 5) -> pd.Series:
        returns = self.get_returns()
        return returns.tail(days)

    def get_moving_average(self, window: int = 20) -> pd.Series:
        return self.data["Close"].rolling(window=window).mean()

    def get_volatility(self, window: int = 20, log: bool = False) -> pd.Series:
        returns = self.get_returns(log=log)
        return returns.rolling(window).std()

    def get_summary_stats(self) -> Dict[str, float]:
        returns = self.get_returns()
        return {
            "mean_return": returns.mean(),
            "std_dev": returns.std(),
            "sharpe_ratio": (returns.mean() / returns.std()) * np.sqrt(252),
            "max_drawdown": self._calculate_max_drawdown()
        }

    def _calculate_max_drawdown(self) -> float:
        cumulative = (1 + self.get_returns()).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def get_fundamentals(self) -> Dict[str, Optional[float]]:
        ticker_obj = yf.Ticker(self.ticker)
        info = ticker_obj.info
        keys = ["trailingPE", "forwardPE", "priceToBook",
                "marketCap", "dividendYield", "beta"]
        return {k: info.get(k, None) for k in keys}

    def refresh(self):
        self.data = self._download_data()


# Test block
if __name__ == "__main__":
    print("[Testing YFinanceFetcher...]\n")

    yf_fetcher = YFinanceFetcher("AAPL", period="6mo", interval="1d")

    print("\n[Summary Stats]")
    print(yf_fetcher.get_summary_stats())

    print("\n[Recent Returns]")
    print(yf_fetcher.get_returns().tail())

    print("\n[Fundamentals]")
    print(yf_fetcher.get_fundamentals())

    print("\n[20-day Moving Average]")
    print(yf_fetcher.get_moving_average().tail())
