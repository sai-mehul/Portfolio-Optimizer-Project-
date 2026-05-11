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


###for testing
rets = np.array([-0.02, 0.015, -0.04, 0.03, -0.01, 0.005, -0.03, 0.01])
