import numpy as np
import pandas as pd
from yfinance_fetcher import YFinanceFetcher
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class AlphaModel:
    def __init__(self):
        self.fetcher = YFinanceFetcher(
            ticker="AAPL", period="1y", interval="1d")
        raw = self.fetcher.get_price_data().copy()

        raw.columns.name = None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        self.data = raw

    def momentum_strategy(self, window=10):
        print("[AlphaModel] Running Momentum Strategy...")
        df = self.data.copy().reset_index()
        df["momentum"] = df["Close"] - df["Close"].shift(window)
        df["signal_momentum"] = np.where(df["momentum"] > 0, 1, -1)
        return df[["Close", "momentum", "signal_momentum"]].dropna()

    def mean_reversion_strategy(self, window=10):
        print("[AlphaModel] Running Mean Reversion Strategy...")
        df = self.data.copy().reset_index()
        df["rolling_mean"] = df["Close"].rolling(window=window).mean()
        df["rolling_std"] = df["Close"].rolling(window=window).std()
        df["z_score"] = (df["Close"] - df["rolling_mean"]) / df["rolling_std"]
        df["signal_meanrev"] = np.where(df["z_score"] > 1, -1,
                                        np.where(df["z_score"] < -1, 1, 0))
        return df[["Close", "z_score", "signal_meanrev"]].dropna()

    def moving_average_crossover(self, short_window=5, long_window=20):
        print("[AlphaModel] Running Moving Average Crossover Strategy...")
        df = self.data.copy().reset_index()
        df["short_ma"] = df["Close"].rolling(window=short_window).mean()
        df["long_ma"] = df["Close"].rolling(window=long_window).mean()
        df["signal_mac"] = np.where(df["short_ma"] > df["long_ma"], 1, -1)
        return df[["Close", "short_ma", "long_ma", "signal_mac"]].dropna()

    def factor_model(self):
        print("[AlphaModel] Running Simple Factor Model...")
        df = self.data.copy().reset_index()
        df["momentum"] = df["Close"].pct_change(periods=5)
        df["volatility"] = df["Close"].rolling(window=10).std()
        df["factor_score"] = df["momentum"] / df["volatility"]
        df["signal_factor"] = np.where(df["factor_score"] > 0, 1, -1)
        return df[["Close", "factor_score", "signal_factor"]].dropna()

    def machine_learning_model(self):
        print("[AlphaModel] Running ML Model (Random Forest)...")
        df = self.data.copy().reset_index()
        df["returns"] = df["Close"].pct_change()
        df["ma10"] = df["Close"].rolling(window=10).mean()
        df["ma50"] = df["Close"].rolling(window=50).mean()
        df["volatility"] = df["Close"].rolling(window=10).std()
        df["target"] = np.where(df["returns"].shift(-1) > 0, 1, 0)

        df = df[["ma10", "ma50", "volatility", "target"]].dropna()
        X = df[["ma10", "ma50", "volatility"]]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        print(classification_report(y_test, preds))
        return clf


#Test Block
if __name__ == "__main__":
    alpha = AlphaModel()

    print("\n--- Momentum Strategy ---")
    print(alpha.momentum_strategy().tail())

    print("\n--- Mean Reversion Strategy ---")
    print(alpha.mean_reversion_strategy().tail())

    print("\n--- Moving Average Crossover ---")
    print(alpha.moving_average_crossover().tail())

    print("\n--- Factor Model ---")
    print(alpha.factor_model().tail())

    print("\n--- Machine Learning Model ---")
    alpha.machine_learning_model()
