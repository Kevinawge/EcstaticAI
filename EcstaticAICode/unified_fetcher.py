from yfinance_fetcher import YFinanceFetcher
from fred_fetcher import FREDFetcher
from crypto_fetcher import CryptoFetcher

from typing import Optional, Dict, Any, Union
import pandas as pd


class UnifiedFinancialFetcher:
    def __init__(
        self,
        yfinance_ticker: Optional[str] = None,
        fred_series: Optional[str] = None,
        crypto_exchange: str = "coinbase",
        crypto_symbol: str = "BTC/USD",
    ):
        self.yf = YFinanceFetcher(yfinance_ticker or "AAPL")
        self.fred = FREDFetcher()
        self.crypto = CryptoFetcher(crypto_exchange)
        self.crypto_symbol = crypto_symbol

    #STOCK METHODS
    def get_stock_summary(self) -> Dict[str, Any]:
        return self.yf.get_summary_stats()

    def get_stock_moving_average(self, window: int = 20) -> pd.DataFrame:
        return self.yf.get_moving_average(window)

    def get_stock_recent_returns(self, days: int = 5) -> pd.DataFrame:
        return self.yf.get_recent_returns(days)

    def get_stock_fundamentals(self) -> Dict[str, Any]:
        return self.yf.get_fundamentals()

    #FRED METHODS
    def get_macro_data(self, series: str) -> pd.DataFrame:
        return self.fred.fetch_series(series)

    def get_macro_summary(self, series: str) -> Dict[str, float]:
        return self.fred.get_summary_stats(series)

    #CRYPTO METHODS
    def get_crypto_ohlcv(self, limit: int = 30, timeframe: str = "1d") -> pd.DataFrame:
        return self.crypto.fetch_ohlcv(self.crypto_symbol, timeframe, limit)

    def get_crypto_summary(self, limit: int = 30, timeframe: str = "1d") -> Dict[str, float]:
        return self.crypto.get_summary_stats(self.crypto_symbol, timeframe, limit)

    def get_crypto_price(self) -> float:
        return self.crypto.get_latest_price(self.crypto_symbol)

    def get_crypto_symbols(self) -> list:
        return self.crypto.get_supported_symbols()


#Test block
if __name__ == "__main__":
    print("[Testing UnifiedFinancialFetcher...]\n")

    unified = UnifiedFinancialFetcher(
        yfinance_ticker="AAPL", crypto_symbol="BTC/USD")

    #Stock Data
    print("[Stock Summary: AAPL]")
    stock_summary = unified.get_stock_summary()
    print(stock_summary)

    print("\n[20-day Moving Average: AAPL]")
    ma = unified.get_stock_moving_average(20)
    print(ma.tail())

    print("\n[Recent Returns: AAPL]")
    print(unified.get_stock_recent_returns(5))

    print("\n[Fundamentals: AAPL]")
    print(unified.get_stock_fundamentals())

    #FRED Data
    print("\n[Macro Series: CPI (FRED/CPIAUCSL)]")
    cpi_data = unified.get_macro_data("CPIAUCSL")
    print(cpi_data.tail())

    print("\n[Macro Summary: CPI]")
    print(unified.get_macro_summary("CPIAUCSL"))

    #Crypto Data
    print("\n[Crypto OHLCV: BTC/USD]")
    ohlcv = unified.get_crypto_ohlcv()
    print(ohlcv.tail())

    print("\n[Crypto Summary: BTC/USD]")
    crypto_summary = unified.get_crypto_summary()
    print(crypto_summary)

    print("\n[Latest Crypto Price: BTC/USD]")
    print(unified.get_crypto_price())
