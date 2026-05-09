import pandas as pd
import numpy as np


class PortfolioOptimizer:
    
    def __init__(self, price_data, risk_free_rate=0.05, sims=10000):
        self.price_data = price_data
        self.risk_free_rate = risk_free_rate
        self.sims = sims
        

        self.returns = price_data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        self.n_assets = len(self.returns.columns)
        self.tickers = list(self.returns.columns)
        

        self.equal_weights = np.array([1/self.n_assets] * self.n_assets)
    
    def _portfolio_stats(self, weights):
        """Return (return, volatility, sharpe) for given weights."""
        ret = np.dot(weights, self.mean_returns)
        vol = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe
    

    def max_sharpe(self):
        """Find portfolio that maximizes Sharpe ratio."""
        sr_best = (np.dot(self.equal_weights, self.mean_returns) - self.risk_free_rate) / \
                  np.sqrt(self.equal_weights @ self.cov_matrix @ self.equal_weights)
        weights_best = self.equal_weights.copy()
        
        improvements = []
        
        for _ in range(self.sims):
            w = np.random.random(self.n_assets)
            w = w / w.sum()
            
            ret, vol, sharpe = self._portfolio_stats(w)
            
            if sharpe > sr_best:
                sr_best = sharpe
                weights_best = w
                improvements.append({
                    "weights": w.tolist(),
                    "sharpe": sharpe,
                    "return": ret,
                    "volatility": vol
                })
        
        best_ret, best_vol, _ = self._portfolio_stats(weights_best)
        
        return {
            "weights": dict(zip(self.tickers, weights_best)),
            "sharpe": sr_best,
            "return": best_ret,
            "volatility": best_vol,
            "improvements": len(improvements)
        }

    def min_variance(self):
        """Find portfolio with minimum volatility."""
        vol_best = float('inf')
        weights_best = self.equal_weights.copy()
        
        for _ in range(self.sims):
            w = np.random.random(self.n_assets)
            w = w / w.sum()
            
            vol = np.sqrt(w @ self.cov_matrix @ w)
            
            if vol < vol_best:
                vol_best = vol
                weights_best = w
        
        ret_best = np.dot(weights_best, self.mean_returns)
        
        return {
            "weights": dict(zip(self.tickers, weights_best)),
            "volatility": vol_best,
            "return": ret_best
        }
    

    def max_return_for_vol(self, target_vol):
        """Maximize return subject to volatility <= target_vol."""
        ret_best = -float('inf')
        weights_best = self.equal_weights.copy()
        
        for _ in range(self.sims):
            w = np.random.random(self.n_assets)
            w = w / w.sum()
            
            ret, vol, _ = self._portfolio_stats(w)
            
            if vol <= target_vol and ret > ret_best:
                ret_best = ret
                weights_best = w
        
        best_vol = np.sqrt(weights_best @ self.cov_matrix @ weights_best)
        
        return {
            "weights": dict(zip(self.tickers, weights_best)),
            "return": ret_best,
            "volatility": best_vol
        }
    
    def efficient_frontier(self, n_points=20, tolerance=0.0001):
        # Endpoints
        min_var_result = self.min_variance()
        min_var_return = min_var_result["return"]
        
        max_return_value = self.mean_returns.max()
        
        # Target returns
        step = (max_return_value - min_var_return) / (n_points - 1)
        targets = [min_var_return + i * step for i in range(n_points)]
        
        frontier_points = []
        
        for target in targets:
            best_vol = float('inf')
            best_weights = self.equal_weights.copy()
            
            for _ in range(self.sims // n_points):  # split sims across targets
                w = np.random.random(self.n_assets)
                w = w / w.sum()
                
                ret, vol, _ = self._portfolio_stats(w)
                
                if abs(ret - target) < tolerance:
                    if vol < best_vol:
                        best_vol = vol
                        best_weights = w
            
            frontier_points.append({
                "target_return": target,
                "volatility": best_vol,
                "weights": dict(zip(self.tickers, best_weights))
            })
        
        # Max Sharpe point
        max_sharpe_result = self.max_sharpe()
        
        # Individual assets
        individual_assets = []
        for ticker in self.tickers:
            w = np.zeros(self.n_assets)
            w[self.tickers.index(ticker)] = 1.0
            
            individual_assets.append({
                "ticker": ticker,
                "return": self.mean_returns[ticker],
                "volatility": np.sqrt(self.cov_matrix[ticker][ticker]),
                "weights": dict(zip(self.tickers, w))
            })
        
        return {
            "frontier": frontier_points,
            "max_sharpe": max_sharpe_result,
            "individual_assets": individual_assets
        }



############3test data 

price_data = pd.DataFrame({
    'AAPL':  [150.0, 152.0, 151.0, 153.5, 155.0],
    'GOOGL': [140.0, 141.5, 143.0, 142.0, 144.0],
    'JPM':   [155.0, 153.0, 157.0, 156.5, 158.0]
}, index=pd.date_range('2024-01-01', periods=5, freq='B'))





