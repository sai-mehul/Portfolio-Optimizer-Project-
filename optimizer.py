import pandas as pd
import numpy as np

def Sharpe_with_weight(risk_free_rate,price_data,sims):
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()


    init_weights = []

    for i in range(len(price_data.columns)):
        init_weights.append(1/len(price_data.columns))

    df = pd.DataFrame({"Weights":pd.Series(dtype="object"),"Sharpe":pd.Series(dtype="float")})

    pv = np.sqrt(np.transpose(init_weights)@cov_matrix@init_weights)
    pr = np.dot(init_weights,mean_returns)
    sr = (pr-risk_free_rate)/pv

    for i in range(sims):
        adjusted_weights = np.random.random(len(returns.columns))
        adjusted_weights = adjusted_weights/adjusted_weights.sum()

        n_pv = np.sqrt(np.transpose(adjusted_weights)@cov_matrix@adjusted_weights)
        n_pr = np.dot(adjusted_weights,mean_returns)
        n_sr = (n_pr - risk_free_rate)/n_pv

        if n_sr > sr:
            sr = n_sr
            weights = adjusted_weights
            df.loc[len(df)] = [adjusted_weights.tolist(), n_sr]



    best_portfolio = df.loc[df["Sharpe"].idxmax()]

    return best_portfolio


def port_variance(price_data,sims):
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    init_weights = []

    for i in range(len(price_data.columns)):
        init_weights.append(1/len(price_data.columns))


    port_variance = np.transpose(init_weights)@cov_matrix@init_weights

    port_volatility = np.sqrt(port_variance)

    best_volatility = float('inf')
    best_return = 0

    for i in range(sims):
        nw = np.random.random(len(returns.columns))
        nw = nw/nw.sum()

        new_volatility = np.sqrt(np.transpose(nw)@cov_matrix@nw)

        if new_volatility < port_volatility:
            best_volatility = new_volatility
            init_weights = nw
            best_return = np.dot(nw,mean_returns)
    return{"volatility": best_volatility,"weights": init_weights,"best possible returns":best_return}





def efficient_frontier(price_data, risk_free_rate, n_points, sims,tolerance=0.0001):
    returns = price_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    data_for_use = port_variance(price_data,sims)

    volatility = data_for_use["Volatility"]
    best_possible_return_with_low_vol = data_for_use["best possible returns"]

    max_return_asset = mean_retruns.idxmax()
    max_retrun_values = mean_retruns.max()

    max_returns_weight = [0]*len(returns.columns)
    asset_index = list(returns.columns).index(max_returns)
    max_return_weight[asset_index] = 1.0

    step_size = (max_return_value - best_possible_return_with_low_vol)/(n_points-1)
    
    targets = []
    frontier_points = []

    for i in range(n_points):
        target = min_vol_retrun + i*step_size
        targets.append(target)

    weights = []

    for i in range(len(returns.columns)):
        weights.append(1/len(returns.columns))

    for target in targets:
        best_vol_for_target = np.inf()
        best_weights_for_target = weights

        for i in range(sims):
            nw = np.random.random(len(returns.columns))
            nw = nw/nw.sum()

            n_pr = np.dot(weights,mean_returns)
            n_pv = np.sqrt(np.transpose(nw)@cov@nw)

            if abs(n_pr - target)<tolerance:

                if n_pv < best_vol_for_target:
                    best_vol_for_target = n_pv
                    best_weights_for_target = nw

        frontier_points.append([target,best_vol,best_weights_for_target])

############3test data 

price_data = pd.DataFrame({
    'AAPL':  [150.0, 152.0, 151.0, 153.5, 155.0],
    'GOOGL': [140.0, 141.5, 143.0, 142.0, 144.0],
    'JPM':   [155.0, 153.0, 157.0, 156.5, 158.0]
}, index=pd.date_range('2024-01-01', periods=5, freq='B'))


