import pandas_datareader.data as web
import pandas as pd
import datetime
from typing import Optional, Union, List, Dict


class FREDFetcher:
    def __init__(self, start_date: Union[str, datetime.date] = "2010-01-01", end_date: Optional[Union[str, datetime.date]] = None):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(
            end_date) if end_date else datetime.datetime.today()
        self.series_cache: Dict[str, pd.Series] = {}

    def fetch_series(self, series_id: str) -> pd.Series:
        if series_id in self.series_cache:
            return self.series_cache[series_id]

        try:
            data = web.DataReader(
                series_id, "fred", self.start_date, self.end_date)
            series = data[series_id].dropna()
            self.series_cache[series_id] = series
            return series
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch FRED series '{series_id}': {str(e)}")

    def get_latest_value(self, series_id: str) -> float:
        series = self.fetch_series(series_id)
        return series.iloc[-1]

    def get_change_over_period(self, series_id: str, periods: int = 12) -> float:
        series = self.fetch_series(series_id)
        if len(series) < periods:
            raise ValueError(
                f"Not enough data to calculate change over {periods} periods")
        return (series.iloc[-1] - series.iloc[-periods]) / series.iloc[-periods]

    def get_rolling_average(self, series_id: str, window: int = 3) -> pd.Series:
        series = self.fetch_series(series_id)
        return series.rolling(window).mean()

    def get_summary_stats(self, series_id: str) -> Dict[str, float]:
        series = self.fetch_series(series_id)
        return {
            "mean": series.mean(),
            "std_dev": series.std(),
            "min": series.min(),
            "max": series.max(),
            "latest": series.iloc[-1]
        }

    def get_multiple_series(self, series_ids: List[str]) -> pd.DataFrame:
        data = {}
        for sid in series_ids:
            try:
                data[sid] = self.fetch_series(sid)
            except Exception as e:
                print(f"[Warning] Failed to fetch {sid}: {e}")
        return pd.DataFrame(data)

    def refresh_cache(self):
        self.series_cache.clear()


# Test block
if __name__ == "__main__":
    print("[Testing FREDFetcher...]\n")

    fred = FREDFetcher(start_date="2015-01-01")

    series_id = "CPIAUCSL"

    print(f"\n[Latest Value for {series_id}]")
    print(fred.get_latest_value(series_id))

    print(f"\n[Summary Stats for {series_id}]")
    print(fred.get_summary_stats(series_id))

    print(f"\n[12-month Change for {series_id}]")
    change = fred.get_change_over_period(series_id, periods=12)
    print(f"{change * 100:.2f}%")

    print(f"\n[3-month Moving Average for {series_id}]")
    print(fred.get_rolling_average(series_id, window=3).tail())

    # Fetch multiple macro indicators
    print("\n[Multiple Series Fetch: CPI, UNRATE, FEDFUNDS]")
    df = fred.get_multiple_series(["CPIAUCSL", "UNRATE", "FEDFUNDS"])
    print(df.tail())
