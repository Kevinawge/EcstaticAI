import ccxt
import pandas as pd
import time
from typing import List, Dict, Optional


class CryptoFetcher:
    def __init__(self, exchange_name: str = "coinbase", rate_limit: float = 1.5):
        self.exchange_name = exchange_name.lower()
        self.exchange = self._load_exchange()
        self.rate_limit = rate_limit
        self.last_call = 0.0

    def _load_exchange(self):
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class({'enableRateLimit': True})
            if not exchange.has.get("fetchOHLCV", False):
                raise ValueError(
                    f"{self.exchange_name} does not support OHLCV.")
            exchange.load_markets()
            return exchange
        except Exception as e:
            raise RuntimeError(
                f"Failed to load exchange {self.exchange_name}: {str(e)}")

    def _throttle(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_call = time.time()

    def get_supported_symbols(self) -> List[str]:
        return list(self.exchange.symbols)

    def fetch_ohlcv(self, symbol: str = "BTC/USD", timeframe: str = "1d", limit: int = 90) -> pd.DataFrame:
        self._throttle()
        if symbol not in self.exchange.symbols:
            raise ValueError(
                f"Symbol {symbol} not supported by {self.exchange_name}")

        ohlcv = self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    def get_latest_price(self, symbol: str = "BTC/USD") -> float:
        self._throttle()
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker["last"]

    def get_summary_stats(self, symbol: str = "BTC/USD", timeframe: str = "1d", limit: int = 90) -> Dict[str, float]:
        df = self.fetch_ohlcv(symbol, timeframe, limit)
        return {
            "mean_close": df["close"].mean(),
            "max_close": df["close"].max(),
            "min_close": df["close"].min(),
            "volatility": df["close"].std(),
            "latest_close": df["close"].iloc[-1]
        }

    def get_exchange_name(self) -> str:
        return self.exchange.name


# Test block
if __name__ == "__main__":
    print("[Testing CryptoFetcher with Coinbase...]\n")

    fetcher = CryptoFetcher("coinbase")

    print("[Exchange Info]")
    print("Name:", fetcher.get_exchange_name())

    print("\n[Symbols Supported]")
    symbols = fetcher.get_supported_symbols()
    print(f"{len(symbols)} symbols loaded. Showing first 10:")
    print(symbols[:10])

    print("\n[Latest Price: BTC/USD]")
    print(fetcher.get_latest_price("BTC/USD"))

    print("\n[OHLCV Data: BTC/USD - 1d]")
    ohlcv_df = fetcher.fetch_ohlcv("BTC/USD", "1d", limit=30)
    print(ohlcv_df.tail())

    print("\n[Summary Stats: BTC/USD]")
    stats = fetcher.get_summary_stats("BTC/USD")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")
