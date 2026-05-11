import numpy as np
from scipy.stats import norm

class RiskMetrics:
    def __init__(self, returns, portfolio_value=100000):
        self.returns = np.asarray(returns)
        self.portfolio_value = portfolio_value

    def var_historical(self, alpha=0.05):
        sorted_rets = np.sort(self.returns)
        idx = int(alpha * len(sorted_rets))
        return self.portfolio_value * sorted_rets[idx]

    def var_parametric(self, alpha=0.05):
        mu = np.mean(self.returns)
        sigma = np.std(self.returns, ddof=0)
        z = norm.ppf(alpha)
        return self.portfolio_value * (z * sigma - mu)

    def var_monte_carlo(self, alpha=0.05, n_sims=10000):
        mu = np.mean(self.returns)
        sigma = np.std(self.returns, ddof=0)
        sim_returns = np.random.normal(mu, sigma, n_sims)
        sorted_sim = np.sort(sim_returns)
        idx = int(alpha * len(sorted_sim))
        return self.portfolio_value * sorted_sim[idx]

    def cvar(self, alpha=0.05):
        sorted_rets = np.sort(self.returns)
        idx = int(alpha * len(sorted_rets))
        tail = sorted_rets[:idx+1]
        cvar_return = np.mean(tail)
        return self.portfolio_value * cvar_return

    def backtest(self, window=252, alpha=0.05):
        breaches = 0
        n_tests = len(self.returns) - window
        for i in range(window, len(self.returns)):
            window_returns = self.returns[i-window:i]
            sorted_win = np.sort(window_returns)
            var_idx = int(alpha * len(sorted_win))
            var_est = sorted_win[var_idx]
            actual = self.returns[i]
            if actual < var_est:
                breaches += 1
        expected = alpha * n_tests
        return {
            "breaches": breaches,
            "expected": expected,
            "ratio": breaches / n_tests if n_tests > 0 else 0
        }
