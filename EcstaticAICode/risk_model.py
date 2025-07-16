import numpy as np
import pandas as pd
from typing import Dict, Union, Optional


class RiskModel:
    def __init__(self, returns: Union[pd.Series, pd.DataFrame], risk_free_rate: float = 0.02):
        if isinstance(returns, pd.Series):
            self.returns = returns.to_frame(name="Asset")
        else:
            self.returns = returns.copy()

        self.risk_free_rate = risk_free_rate

    def sharpe_ratio(self) -> pd.Series:
        excess_returns = self.returns.sub(self.risk_free_rate / 252)
        return excess_returns.mean() / excess_returns.std()

    def beta(self, market_returns: pd.Series) -> pd.Series:
        betas = {}
        for asset in self.returns.columns:
            covariance = np.cov(self.returns[asset], market_returns)[0][1]
            market_var = np.var(market_returns)
            betas[asset] = covariance / market_var
        return pd.Series(betas)

    def alpha(self, market_returns: pd.Series) -> pd.Series:
        alphas = {}
        betas = self.beta(market_returns)
        market_mean = market_returns.mean()
        for asset in self.returns.columns:
            expected = self.risk_free_rate / 252 + \
                betas[asset] * (market_mean - self.risk_free_rate / 252)
            alphas[asset] = self.returns[asset].mean() - expected
        return pd.Series(alphas)

    def value_at_risk(self, confidence_level: float = 0.95) -> pd.Series:
        return self.returns.quantile(1 - confidence_level)

    def expected_shortfall(self, confidence_level: float = 0.95) -> pd.Series:
        return self.returns[self.returns.lt(self.value_at_risk(confidence_level))].mean()

    def max_drawdown(self) -> pd.Series:
        drawdowns = {}
        for asset in self.returns.columns:
            cum_returns = (1 + self.returns[asset]).cumprod()
            peak = cum_returns.cummax()
            dd = (cum_returns - peak) / peak
            drawdowns[asset] = dd.min()
        return pd.Series(drawdowns)

    def capm(self, market_returns: pd.Series) -> pd.DataFrame:
        betas = self.beta(market_returns)
        alphas = self.alpha(market_returns)
        return pd.DataFrame({"Alpha": alphas, "Beta": betas})


# Test block
if __name__ == "__main__":
    print("[Testing RiskModel...]")

    np.random.seed(42)
    asset_returns = pd.DataFrame({
        "AAPL": np.random.normal(0.001, 0.02, 252),
        "MSFT": np.random.normal(0.0008, 0.018, 252),
    })
    market_returns = pd.Series(np.random.normal(
        0.001, 0.015, 252), name="Market")

    model = RiskModel(asset_returns, risk_free_rate=0.02)

    print("\n[Sharpe Ratios]")
    print(model.sharpe_ratio())

    print("\n[Beta vs Market]")
    print(model.beta(market_returns))

    print("\n[Alpha vs Market]")
    print(model.alpha(market_returns))

    print("\n[CAPM Table]")
    print(model.capm(market_returns))

    print("\n[Max Drawdown]")
    print(model.max_drawdown())

    print("\n[Value at Risk (95%)]")
    print(model.value_at_risk())

    print("\n[Expected Shortfall (95%)]")
    print(model.expected_shortfall())
