import numpy as np
import pandas as pd
from scipy.stats import norm 

def hist_var(returns,alpha,portfolio_val):
    sorted_rets = np.sort(returns)
    index_of_VaR = int(alpha*len(sorted_rets))

    VaR_amount = portfolio_val*sorted_rets[index_of_VaR]

    return VaR_amount


def parametric_var(returns,alpha,portfolio_value):
    mu = np.mean(returns)
    sigma = np.std(returns)

    z = norm.ppf(alpha)

    Var = portfolio_value*(z*sigma-mu)

    return Var

def MC_Var(returns,alpha,portfolio_value,n_rets):
    mu = np.mean(returns)
    sigma = np.std(returns)

    dist = np.random.normal(mu,sigma,n_rets)
    dist_sorted = np.sort(dist)

    alpha_percentile = int(alpha*len(dist_sorted))

    VaR = portfolio_value*dist_sorted[alpha_percentile]

    return VaR

###for testing
rets = np.array([-0.02, 0.015, -0.04, 0.03, -0.01, 0.005, -0.03, 0.01])
